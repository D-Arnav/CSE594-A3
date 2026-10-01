"""Batch inference -> raw_outputs.jsonl + {split}_dataset.csv (SPEC 5.5, 6), and the JSON baseline (5.4).

python src/predict.py --split dev --config config.yaml --fit-temperature
python src/predict.py --split study --config config.yaml
python src/predict.py --split dev --config config.yaml --baseline-json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, OUTPUTS, ROOT, countries, load_config, seed_everything  # noqa: E402
from model import (TEMPERATURE_FILE, apply_temperature, candidate_order, fit_temperature,  # noqa: E402
                   load_temperature, predict_record)
from prompts import render_json_baseline  # noqa: E402

COLUMNS = ["trial_id", "image_path", "source_id", "country", "cluster", "has_text",
           "ai_top1", "ai_top2", "ai_top3", "ai_p1", "ai_p2", "ai_p3",
           "ai_rationale1", "ai_rationale2", "ai_rationale3",
           "ai_correct_top1", "ai_correct_top3", "gt_rank",
           "error_category", "rationale_faithful_top1", "human_catchable",
           "model_id", "revision", "prompt_version", "temperature", "max_pixels"]


def out_dir(split):
    return OUTPUTS if split == "study" else OUTPUTS / split


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path, records):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(path)


def load_existing(path, cfg, overwrite=False):
    """Records already in the jsonl, for resuming. Refuses if they came from another model/prompt."""
    records = read_jsonl(path)
    want = (cfg["model"]["model_id"], cfg["model"]["revision"], dict(cfg["inference"]["prompt_versions"]))
    stale = [r["trial_id"] for r in records if (r["model_id"], r["revision"], r["prompt_versions"]) != want]
    if stale and not overwrite:
        sys.exit(f"{path} has {len(stale)} records (e.g. {stale[0]}) from a different model_id/revision/"
                 f"prompt_versions than the config. Re-run with --overwrite to start over.")
    if stale:
        Path(path).unlink()
        return []
    return records


def prompt_version_str(pv):
    return ";".join(f"{k}={v}" for k, v in pv.items())


def gt_rank(logprobs, country):
    """Rank of the true country among all candidates by logprob (1 = best), same tie order as top-3."""
    ranked = sorted(logprobs, key=lambda c: -logprobs[c])
    return ranked.index(country) + 1 if country in ranked else ""


def dataset_row(rec, m):
    """One SPEC 6 row from a jsonl record and its meta row (dict)."""
    top = rec["top3"]
    guesses = [t["country"] for t in top]
    row = {k: m[k] for k in ["trial_id", "image_path", "source_id", "country", "cluster", "has_text"]}
    for i in range(3):
        t = top[i] if i < len(top) else {"country": "", "p": "", "rationale": ""}
        row[f"ai_top{i + 1}"], row[f"ai_p{i + 1}"], row[f"ai_rationale{i + 1}"] = t["country"], t["p"], t["rationale"]
    row.update(ai_correct_top1=bool(guesses) and guesses[0] == m["country"],
               ai_correct_top3=m["country"] in guesses,
               gt_rank=gt_rank(rec["logprobs"], m["country"]),
               error_category="", rationale_faithful_top1="", human_catchable="",
               model_id=rec["model_id"], revision=rec["revision"],
               prompt_version=prompt_version_str(rec["prompt_versions"]),
               temperature=rec["temperature"], max_pixels=rec.get("max_pixels"))
    return {k: row[k] for k in COLUMNS}


def write_dataset(records, meta, path):
    by_id = {r["trial_id"]: r for r in records}
    rows = [dataset_row(by_id[m["trial_id"]], m) for m in meta.to_dict("records") if m["trial_id"] in by_id]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False)
    return df


def load_image(rel_path):
    from PIL import Image
    return Image.open(ROOT / rel_path).convert("RGB")


# ---- SPEC 5.4 generative JSON baseline ------------------------------------------
ALIASES = {"usa": "United States", "us": "United States", "united states of america": "United States",
           "uk": "United Kingdom", "great britain": "United Kingdom", "britain": "United Kingdom",
           "england": "United Kingdom", "scotland": "United Kingdom", "wales": "United Kingdom",
           "korea": "South Korea", "republic of korea": "South Korea", "russian federation": "Russia"}


def normalize_country(name, labels):
    """Map a free-text country to the label set, or None if it is outside it."""
    key = str(name or "").strip().strip(".").lower()
    lower = {l.lower(): l for l in labels}
    return lower.get(key) or (ALIASES[key] if ALIASES.get(key) in labels else None)


def parse_guesses(text):
    """Best-effort JSON parse of a baseline answer -> list of {country, confidence, reasoning} or None."""
    s = re.sub(r"```(?:json)?", "", text or "")
    obj = None
    for a, b in ["{}", "[]"]:
        i, j = s.find(a), s.rfind(b)
        if 0 <= i < j:
            try:
                obj = json.loads(s[i:j + 1])
                break
            except json.JSONDecodeError:
                continue
    if isinstance(obj, dict):
        obj = obj.get("guesses", [obj] if "country" in obj else None)
    if not isinstance(obj, list):
        return None
    out = []
    for g in obj:
        if not isinstance(g, dict):
            continue
        try:
            conf = float(g.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        out.append({"country": g.get("country"), "confidence": conf, "reasoning": g.get("reasoning")})
    return out or None


def baseline_record(trial_id, truth, raw, labels, runtime_s, truncated=False):
    guesses = parse_guesses(raw)
    norm = [normalize_country(g["country"], labels) for g in guesses or []]
    return {"trial_id": trial_id, "country": truth, "raw": raw, "truncated": truncated, "parse_ok": guesses is not None,
            "guesses": guesses, "normalized": norm, "runtime_s": round(runtime_s, 2)}


def baseline_summary(recs):
    n = len(recs)
    parsed = [r for r in recs if r["parse_ok"]]
    all_guesses = [c for r in parsed for c in r["normalized"]]
    sums = [sum(g["confidence"] for g in r["guesses"]) for r in parsed
            if all(g["confidence"] is not None for g in r["guesses"])]
    top1 = [bool(r["normalized"]) and r["normalized"][0] == r["country"] for r in parsed]
    top3 = [r["country"] in r["normalized"][:3] for r in parsed]
    rate = lambda k, d: round(k / d, 4) if d else None  # noqa: E731
    return {"n": n, "n_parsed": len(parsed), "parse_failure_rate": rate(n - len(parsed), n),
            "truncated_rate": rate(sum(r.get("truncated", False) for r in recs), n),
            "not_three_guesses_rate": rate(sum(len(r["guesses"]) != 3 for r in parsed), len(parsed)),
            "out_of_label_set_rate": rate(sum(c is None for c in all_guesses), len(all_guesses)),
            "missing_confidence_rate": rate(len(parsed) - len(sums), len(parsed)),
            "confidence_sum_mean": round(sum(sums) / len(sums), 4) if sums else None,
            "confidence_sum_within_0.05_of_1": rate(sum(abs(s - 1) <= 0.05 for s in sums), len(sums)),
            "confidence_sum_above_1.05": rate(sum(s > 1.05 for s in sums), len(sums)),
            "top1_acc_parsed": rate(sum(top1), len(parsed)), "top3_acc_parsed": rate(sum(top3), len(parsed)),
            "top1_acc_overall": rate(sum(top1), n), "top3_acc_overall": rate(sum(top3), n)}


def run_baseline(gm, cfg, meta, labels, split):
    path = out_dir(split) / "baseline_json.jsonl"
    done = {r["trial_id"] for r in read_jsonl(path)}
    max_new = cfg["inference"].get("baseline_max_new_tokens", 512)
    with open(path, "a") as f:
        for m in tqdm([m for m in meta.to_dict("records") if m["trial_id"] not in done], desc="baseline"):
            t0 = time.time()
            prompt = render_json_baseline(candidate_order(labels, cfg["seed"], m["trial_id"]))
            raw = gm.generate(load_image(m["image_path"]), prompt, max_new)
            truncated = len(gm.tokenizer(raw)["input_ids"]) >= max_new  # hit the cap: likely cut-off JSON
            rec = baseline_record(m["trial_id"], m["country"], raw, labels, time.time() - t0, truncated)
            rec.update(model_id=gm.model_id, revision=gm.revision)
            f.write(json.dumps(rec) + "\n")
            f.flush()
    ids = set(meta["trial_id"])
    summary = baseline_summary([r for r in read_jsonl(path) if r["trial_id"] in ids])
    summary.update(model_id=gm.model_id, revision=gm.revision, prompt_version="json=v1", split=split)
    (out_dir(split) / "baseline_json_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["study", "dev"], required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--fit-temperature", action="store_true", help="fit T on dev, write outputs/temperature.json")
    ap.add_argument("--limit", type=int, help="only the first N trials of the split")
    ap.add_argument("--baseline-json", action="store_true", help="run the SPEC 5.4 JSON baseline instead")
    ap.add_argument("--naive", action="store_true", help="score without prefix KV caching (slow reference)")
    ap.add_argument("--overwrite", action="store_true", help="discard a jsonl from another model/prompt")
    args = ap.parse_args()
    if args.fit_temperature and args.split != "dev":
        sys.exit("--fit-temperature is only allowed on the dev split")

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    labels = countries(cfg)
    meta = pd.read_csv(DATA / args.split / f"{args.split}_meta.csv", dtype={"trial_id": str, "source_id": str},
                       keep_default_na=False)
    if args.limit:
        meta = meta.head(args.limit)
    out = out_dir(args.split)
    out.mkdir(parents=True, exist_ok=True)

    from model import GeoModel
    if args.baseline_json:
        return run_baseline(GeoModel(cfg), cfg, meta, labels, args.split)

    raw_path = out / "raw_outputs.jsonl"
    records = load_existing(raw_path, cfg, args.overwrite)
    done = {r["trial_id"] for r in records}
    todo = [m for m in meta.to_dict("records") if m["trial_id"] not in done]
    T, source = load_temperature(cfg)
    print(f"{len(done)} trials already done, {len(todo)} to run; T={T:.4f} from {source}")

    if todo:
        gm = GeoModel(cfg)
        print(f"device={gm.device} dtype={gm.dtype} max_pixels={gm.max_pixels}")
        with open(raw_path, "a") as f:
            for m in tqdm(todo, desc=args.split):
                rec = predict_record(gm, load_image(m["image_path"]), m["trial_id"], cfg, labels, T,
                                     naive=args.naive)
                f.write(json.dumps(rec) + "\n")
                f.flush()

    all_records = read_jsonl(raw_path)
    ids = set(meta["trial_id"])
    records = [r for r in all_records if r["trial_id"] in ids]  # same dicts: edits below land in all_records
    truth = dict(zip(meta["trial_id"], meta["country"]))
    if args.fit_temperature:
        T, nll_before, nll_after = fit_temperature([r["logprobs"] for r in records],
                                                   [truth[r["trial_id"]] for r in records])
        TEMPERATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TEMPERATURE_FILE.write_text(json.dumps({"T": T, "nll_before": nll_before, "nll_after": nll_after,
                                                "n": len(records), "split": "dev",
                                                "model_id": cfg["model"]["model_id"],
                                                "revision": cfg["model"]["revision"]}, indent=2))
        source = "temperature.json"
        print(f"fitted T={T:.4f}  NLL {nll_before:.4f} -> {nll_after:.4f}  (n={len(records)})")
    for r in records:
        apply_temperature(r, T)
        r["temperature_source"] = source
    write_jsonl(raw_path, all_records)

    df = write_dataset(records, meta, out / f"{args.split}_dataset.csv")
    n = len(df)
    print(f"{n} trials -> {out / f'{args.split}_dataset.csv'}")
    if n:
        print(f"top-1 acc {df['ai_correct_top1'].mean():.3f}  top-3 acc {df['ai_correct_top3'].mean():.3f}  "
              f"MRR {(1 / pd.to_numeric(df['gt_rank'])).mean():.3f}")
        rt = pd.DataFrame([{k: r.get(k) for k in ["runtime_s", "score_s", "explain_s"]} for r in records])
        print("mean seconds/trial:", rt.mean().round(2).to_dict())


if __name__ == "__main__":
    main()
