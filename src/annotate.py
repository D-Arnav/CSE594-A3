"""Streamlit annotation tool for top-1 errors (SPEC 8.2).

    streamlit run src/annotate.py -- --pred outputs/study_dataset.csv \
        --out outputs/annotations.csv --rater me [--n-correct 50] [--subset-frac 1.0]

Second rater (agreement subset):
    streamlit run src/annotate.py -- --rater r2 --subset-frac 0.25 --out outputs/annotations_rater2.csv

Any flag can also come from env vars ANNOTATE_PRED / ANNOTATE_OUT / ANNOTATE_RATER /
ANNOTATE_N_CORRECT / ANNOTATE_SUBSET_FRAC / ANNOTATE_SEED (used by the AppTest smoke
test, since AppTest cannot pass script args). Command-line flags win over env vars.
"""
import argparse
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUTS, ROOT, load_config  # noqa: E402

ANN_COLUMNS = ["trial_id", "rater", "error_category", "rationale_faithful_top1",
               "human_catchable", "note", "timestamp"]
CATEGORIES = ["HAL", "TXT", "NBR", "KNW", "CUE"]  # decision rule order
FAITHFUL = ["faithful", "partially", "hallucinated", "none"]
CATCHABLE = ["yes", "maybe", "no"]
OVR_THRESHOLD = 0.6

CATEGORY_HELP = {
    "HAL": "Hallucinated evidence: rationale cites something not in the image.",
    "TXT": "Text misread/ignored: legible text or script points elsewhere and the model missed or misread it.",
    "NBR": "Neighbor confusion: truth and prediction in the same cluster with ambiguous cues.",
    "KNW": "Knowledge gap: correct cue noticed but mapped to the wrong country.",
    "CUE": "Cue-poor image: little distinctive evidence; a human would also struggle.",
}
FAITHFUL_HELP = {
    "faithful": "every visual claim is visible in the image",
    "partially": "at least one claim is visible and at least one is not",
    "hallucinated": "no claim is visible in the image",
    "none": "the rationale says there are no distinctive clues",
}
CATCHABLE_HELP = ("Could an attentive non-expert, given the AI output, reasonably reject the top-1?")


def parse_args(argv=None):
    """Parse flags after '--' (Streamlit passes its own argv), env vars as defaults."""
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    env = os.environ.get
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default=env("ANNOTATE_PRED", str(OUTPUTS / "study_dataset.csv")))
    p.add_argument("--out", default=env("ANNOTATE_OUT", str(OUTPUTS / "annotations.csv")))
    p.add_argument("--rater", default=env("ANNOTATE_RATER", "rater1"))
    p.add_argument("--n-correct", type=int, default=int(env("ANNOTATE_N_CORRECT", 50)))
    p.add_argument("--subset-frac", type=float, default=float(env("ANNOTATE_SUBSET_FRAC", 1.0)))
    p.add_argument("--seed", type=int, default=env("ANNOTATE_SEED"))
    args, _ = p.parse_known_args(argv)
    if args.seed is None:
        args.seed = load_config()["seed"]
    return args


def is_error(row):
    return row["ai_top1"] != row["country"]


def build_queue(pred, n_correct=50, seed=42, subset_frac=1.0):
    """All top-1 errors + seeded sample of n_correct correct trials; optional seeded subset.

    Order: errors first, then correct trials, each by trial_id. The subset keeps that order,
    so a second rater with the same seed gets a subset of the first rater's queue.
    """
    wrong = pred["ai_top1"] != pred["country"]
    errors = sorted(pred.loc[wrong, "trial_id"])
    correct = sorted(pred.loc[~wrong, "trial_id"])
    correct = sorted(random.Random(seed).sample(correct, min(n_correct, len(correct))))
    queue = errors + correct
    if subset_frac < 1.0:
        keep = set(random.Random(seed).sample(queue, round(subset_frac * len(queue))))
        queue = [t for t in queue if t in keep]
    return queue


def validate(row, error):
    """Return a list of problems with an annotation row (empty list = valid)."""
    problems = []
    if row.get("rationale_faithful_top1") not in FAITHFUL:
        problems.append(f"rationale_faithful_top1 must be one of {FAITHFUL}")
    if error:
        if row.get("error_category") not in CATEGORIES:
            problems.append(f"error_category must be one of {CATEGORIES}")
        if row.get("human_catchable") not in CATCHABLE:
            problems.append(f"human_catchable must be one of {CATCHABLE}")
    else:
        if row.get("error_category"):
            problems.append("error_category must be blank for a correct trial")
        if row.get("human_catchable"):
            problems.append("human_catchable must be blank for a correct trial")
    return problems


def load_annotations(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=ANN_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)[ANN_COLUMNS]


def upsert(path, row):
    """Insert or replace the row keyed by (trial_id, rater); stamps an ISO timestamp."""
    row = {c: row.get(c) or "" for c in ANN_COLUMNS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ann = load_annotations(path)
    same = (ann["trial_id"] == row["trial_id"]) & (ann["rater"] == row["rater"])
    ann = pd.concat([ann[~same], pd.DataFrame([row])], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ann.to_csv(path, index=False)
    return ann


def done_ids(ann, rater):
    return set(ann.loc[ann["rater"] == rater, "trial_id"])


def progress(queue, ann, rater):
    """(n annotated in queue, queue length)."""
    done = done_ids(ann, rater)
    return sum(t in done for t in queue), len(queue)


def first_unannotated(queue, ann, rater):
    """Index of the first queue item this rater has not annotated (len(queue) if all done)."""
    done = done_ids(ann, rater)
    return next((i for i, t in enumerate(queue) if t not in done), len(queue))


def _image_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _radio(st, label, options, current, key, help=None, captions=None):
    index = options.index(current) if current in options else None
    return st.radio(label, options, index=index, key=key, help=help, captions=captions,
                    horizontal=captions is None)


def main():
    import streamlit as st

    args = parse_args()
    st.set_page_config(page_title="GeoGuess-HAI annotation", layout="wide")
    pred = pd.read_csv(args.pred, dtype={"trial_id": str})
    by_id = pred.set_index("trial_id")
    queue = build_queue(pred, args.n_correct, args.seed, args.subset_frac)
    ann = load_annotations(args.out)
    if not queue:
        st.warning("Queue is empty.")
        return

    if "idx" not in st.session_state:  # resume
        st.session_state.idx = min(first_unannotated(queue, ann, args.rater), len(queue) - 1)

    done, total = progress(queue, ann, args.rater)
    st.progress(done / total, text=f"{done}/{total} annotated by {args.rater} -> {args.out}")
    nav = st.columns(3)
    if nav[0].button("< Prev", key="prev"):
        st.session_state.idx = max(st.session_state.idx - 1, 0)
    if nav[1].button("Next >", key="next"):
        st.session_state.idx = min(st.session_state.idx + 1, total - 1)
    if nav[2].button("Jump to first unannotated", key="jump"):
        st.session_state.idx = min(first_unannotated(queue, ann, args.rater), total - 1)

    tid = queue[st.session_state.idx]
    row = by_id.loc[tid]
    error = is_error(row)
    mine = ann[(ann["trial_id"] == tid) & (ann["rater"] == args.rater)]
    prev = mine.iloc[0].to_dict() if len(mine) else {}

    st.subheader(f"{tid} ({st.session_state.idx + 1}/{total})"
                 + (" - already annotated" if prev else ""))
    left, right = st.columns([3, 2])
    left.image(str(_image_path(row["image_path"])), width="stretch")
    with right:
        st.markdown(f"**Ground truth:** {row['country']} ({row['cluster']})")
        st.markdown("**AI top-1 is " + ("WRONG**" if error else "correct**"))
        if error and float(row["ai_p1"]) >= OVR_THRESHOLD:
            st.error(f"OVR: overconfident miss (p1 = {float(row['ai_p1']):.2f} >= {OVR_THRESHOLD})")
        for k in (1, 2, 3):
            with st.container(border=True):
                mark = " (truth)" if row[f"ai_top{k}"] == row["country"] else ""
                st.markdown(f"**#{k} {row[f'ai_top{k}']}{mark}** - p = {float(row[f'ai_p{k}']):.2f}")
                st.caption(str(row[f"ai_rationale{k}"]))

    with st.expander("Decision rule: first that applies, HAL -> TXT -> NBR -> KNW -> CUE",
                     expanded=error):
        for c in CATEGORIES:
            st.markdown(f"- **{c}**: {CATEGORY_HELP[c]}")

    new = {"trial_id": tid, "rater": args.rater}
    if error:
        new["error_category"] = _radio(st, "Primary error category", CATEGORIES,
                                       prev.get("error_category"), f"cat_{tid}",
                                       captions=[CATEGORY_HELP[c] for c in CATEGORIES])
    new["rationale_faithful_top1"] = _radio(
        st, "Top-1 rationale faithfulness", FAITHFUL, prev.get("rationale_faithful_top1"),
        f"faith_{tid}", captions=[FAITHFUL_HELP[f] for f in FAITHFUL])
    if error:
        new["human_catchable"] = _radio(st, "Human-catchable?", CATCHABLE,
                                        prev.get("human_catchable"), f"catch_{tid}",
                                        help=CATCHABLE_HELP)
    new["note"] = st.text_input("Note", value=prev.get("note", ""), key=f"note_{tid}")

    if st.button("Save", key="save", type="primary"):
        problems = validate(new, error)
        if problems:
            for msg in problems:
                st.error(msg)
        else:
            upsert(args.out, new)
            st.session_state.idx = min(st.session_state.idx + 1, total - 1)
            st.rerun()


if __name__ == "__main__":
    main()
