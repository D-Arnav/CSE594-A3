"""AI assistance: SmolVLM2 ranks the 5 countries for a photo and explains its top 3.

python model.py study   # score every study photo, write data/study_dataset.csv
python model.py S017    # print one trial (evidence that the model runs)
"""
import copy
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID, REVISION = "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4"
COUNTRIES = ["United States", "Germany", "Japan", "South Korea", "India"]
SEED = 42
# Confidence temperature (Guo et al. 2017), fitted by minimizing NLL on a separate set of 200 calibration photos.
T = 0.5903211083205333
RANK = ("You are shown a street-level photo. It was taken in exactly one of these countries:\n"
        "{countries}\nWhich country was it taken in? Answer with the country name only.")
EXPLAIN = ("Look at this street-level photo. In one or two short sentences, describe the specific things you "
           "can see that suggest whether it was taken in {country}.")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA = Path(__file__).parent / "data"


def load():
    proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    proc.image_processor.do_image_splitting = False  # one 384 px view per photo, no tiles
    dtype = torch.float32 if DEVICE == "cpu" else torch.float16
    vlm = AutoModelForImageTextToText.from_pretrained(MODEL_ID, revision=REVISION, dtype=dtype)
    return proc, vlm.to(DEVICE).eval()


def encode(proc, image, prompt, answer=None):
    msgs = [{"role": "user", "content": ([{"type": "image", "image": image}] if image is not None else [])
             + [{"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return proc.apply_chat_template(msgs, add_generation_prompt=answer is None, tokenize=True,
                                    return_dict=True, return_tensors="pt")


def answer_tokens(proc, prompt, answer):
    """Tokens the assistant turn adds after the generation prompt: the answer plus its end-of-turn token."""
    p = encode(proc, None, prompt)["input_ids"][0]
    f = encode(proc, None, prompt, answer)["input_ids"][0]
    assert f[:len(p)].tolist() == p.tolist(), "generation prompt is not a token prefix of the full conversation"
    tail = f[len(p):].tolist()
    while tail and not proc.tokenizer.decode(tail[-1:]).strip():  # drop trailing whitespace tokens
        tail.pop()
    return tail


@torch.no_grad()
def score(proc, vlm, image, prompt):
    """log p(answer = country + end-of-turn) for every country: the image + prompt prefix runs once,
    then all 5 answers are scored as one padded batch on copies of its KV cache."""
    tails = [answer_tokens(proc, prompt, c) for c in COUNTRIES]
    out = vlm(**encode(proc, image, prompt).to(DEVICE), use_cache=True, logits_to_keep=1)
    first = out.logits[0, -1].float().log_softmax(-1)
    ids = torch.full((len(tails), max(map(len, tails))), proc.tokenizer.pad_token_id or 0, dtype=torch.long)
    for j, t in enumerate(tails):
        ids[j, :len(t)] = torch.tensor(t)
    ids = ids.to(DEVICE)
    kv = copy.deepcopy(out.past_key_values)
    kv.batch_repeat_interleave(len(tails))
    logp = vlm(input_ids=ids, past_key_values=kv, use_cache=True).logits.float().log_softmax(-1)
    lp = {c: round((first[t[0]] + logp[j, :len(t) - 1].gather(-1, ids[j, 1:len(t), None]).sum()).item(), 4)
          for j, (c, t) in enumerate(zip(COUNTRIES, tails))}
    assert all(map(math.isfinite, lp.values())), f"non-finite logprobs {lp}"
    return lp


@torch.no_grad()
def explain(proc, vlm, image, country):
    ids = encode(proc, image, EXPLAIN.format(country=country)).to(DEVICE)
    out = vlm.generate(**ids, do_sample=False, max_new_tokens=60)
    return clean(proc.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))


# SmolVLM2 boilerplate: "The specific things that suggest the photo was taken in X include ...",
# "... suggest that the photo was taken in X.", "the presence of" before every item, repeated items.
OPENER = re.compile(r"^(?:the )?(?:specific )?things that suggest (?:that )?(?:the|this|it) (?:photo |image )?"
                    r"(?:was taken|was|is) (?:in|on|at) [^,:]*?\b(?:include|are|is)\b:?\s*", re.I)
TRAILER = re.compile(r",?\s*(?:all |which )?suggests? that (?:the|this) (?:photo|image) (?:was|is) taken "
                     r"(?:in|on) [^.]*", re.I)
PRESENCE = re.compile(r"\bthe presence (?:and style )?of\s+", re.I)
SENTENCE = r'(?<=[.!?])\s+(?=[A-Z"])'  # "U.S. roads" stays whole


def clean(text):
    """At most two sentences (a fragment cut off by the token limit is dropped), minus boilerplate and repeats."""
    sentences = re.split(SENTENCE, " ".join((text or "").split()))
    if len(sentences) > 1 and not sentences[-1].endswith((".", "!", "?", '"')):
        sentences.pop()
    out = []
    for sent in re.split(SENTENCE, " ".join(sentences[:2]).strip()):
        sent = PRESENCE.sub("", TRAILER.sub("", OPENER.sub("", sent)))
        end = sent[-1] if sent and sent[-1] in ".!?" else ""
        seen, items = set(), []
        for item in sent.rstrip(".!?").split(", "):
            key = re.sub(r"^and |[.\s]+$", "", item.lower())
            if key not in seen:
                seen.add(key)
                items.append(item)
        sent = ", ".join(items).strip() + end if items else ""
        if sent:
            out.append(sent[0].upper() + sent[1:])
    return " ".join(out) or "No distinctive clues identified."


def softmax(lp, T):
    z = np.array(list(lp.values()), dtype=float) / T
    p = np.exp(z - z.max())
    return dict(zip(lp, (p / p.sum()).tolist()))


def predict(proc, vlm, trial):
    """-> (logprobs, countries ranked best first, rationales for the top 3) for one meta row."""
    order = list(COUNTRIES)
    random.Random(f"{SEED}-{trial['trial_id']}").shuffle(order)  # per-photo candidate order vs. position bias
    image = Image.open(DATA.parent / trial["image_path"]).convert("RGB")
    lp = score(proc, vlm, image, RANK.format(countries="\n".join(order)))
    ranked = sorted(COUNTRIES, key=lambda c: -lp[c])
    return lp, ranked, [explain(proc, vlm, image, c) for c in ranked[:3]]


def row(trial, lp, ranked, why, T):
    p = softmax(lp, T)
    r = {k: trial[k] for k in ("trial_id", "image_path", "source_id", "country")}
    for i in range(3):
        r[f"ai_top{i + 1}"], r[f"ai_p{i + 1}"], r[f"ai_rationale{i + 1}"] = ranked[i], round(p[ranked[i]], 2), why[i]
    return {**r, "ai_correct_top1": ranked[0] == trial["country"], "ai_correct_top3": trial["country"] in ranked[:3],
            "gt_rank": ranked.index(trial["country"]) + 1, "logprobs": json.dumps(lp), "temperature": T,
            "model_id": MODEL_ID, "revision": REVISION}


def main(arg):
    meta = pd.read_csv(DATA / "study/study_meta.csv", dtype=str)
    proc, vlm = load()
    if arg == "study":
        trials = meta.to_dict("records")
        out = []
        for i, t in enumerate(trials, 1):
            out.append(row(t, *predict(proc, vlm, t), T))
            print(f"{i}/{len(trials)}", flush=True)
        df = pd.DataFrame(out)
        df.to_csv(DATA / "study_dataset.csv", index=False)
        print(f"top-1 {df.ai_correct_top1.mean():.3f}  top-3 {df.ai_correct_top3.mean():.3f}")
    else:
        t = meta.set_index("trial_id").loc[arg].to_dict() | {"trial_id": arg}
        r = row(t, *predict(proc, vlm, t), T)
        print(f"{arg}  true country: {t['country']}")
        for i in (1, 2, 3):
            print(f"{i}. {r[f'ai_top{i}']:<14} {r[f'ai_p{i}']:.2f}  {r[f'ai_rationale{i}']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
