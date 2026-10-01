"""VLM wrapper: load, score candidates by likelihood, explain (SPEC 5.1-5.2).

Public API used by predict.py, server.py and finetune_lora.py:
pick_device, GeoModel, softmax_with_temperature, fit_temperature, candidate_order,
load_temperature, apply_temperature, predict_record, clean_rationale, image_key.
"""
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUTS  # noqa: E402
from prompts import RANK_ANSWER, render_explain, render_rank  # noqa: E402

FALLBACK_RATIONALE = "No distinctive clues identified."
TEMPERATURE_FILE = OUTPUTS / "temperature.json"


def pick_device(pref="auto"):
    if pref and pref != "auto":
        return pref
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GeoModel:
    def __init__(self, cfg, device=None, dtype=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        m = cfg["model"]
        self.model_id, self.revision = m["model_id"], m["revision"]
        self.max_pixels = m.get("max_pixels")
        self.batch_size = cfg.get("inference", {}).get("score_batch_size", 5)
        self.device = device or pick_device(m.get("device", "auto"))
        dtype = dtype or m.get("dtype", "float16")
        # ponytail: fp16 on CPU is slow and partly unsupported, so CPU always runs float32.
        self.dtype = torch.float32 if self.device == "cpu" else getattr(torch, dtype)

        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.tokenizer = self.processor.tokenizer
        ip = getattr(self.processor, "image_processor", None)
        if self.max_pixels and ip is not None and hasattr(ip, "max_pixels"):
            # Qwen-family processors resize so that H*W <= size["longest_edge"]; others ignore max_pixels.
            ip.size = {"shortest_edge": min(ip.size["shortest_edge"], self.max_pixels),
                       "longest_edge": self.max_pixels}
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, revision=self.revision, dtype=self.dtype).to(self.device).eval()

    # ---- inputs -------------------------------------------------------------
    def build_messages(self, image, prompt, answer=None):
        content = ([{"type": "image", "image": image}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def _encode(self, messages, add_generation_prompt=False):
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=True,
            return_dict=True, return_tensors="pt")

    def answer_tokens(self, prompt, answer):
        """Token ids the assistant turn adds after the generation prompt: the answer plus the
        end-of-turn token(s), without anything the template appends after them (Qwen: "\\n").
        Text-only conversation: the image only changes the prefix, which we check is shared."""
        p = self._encode(self.build_messages(None, prompt), add_generation_prompt=True)["input_ids"][0]
        f = self._encode(self.build_messages(None, prompt, answer))["input_ids"][0]
        n = len(p)
        assert f[:n].tolist() == p.tolist(), "generation prompt is not a token prefix of the full conversation"
        tail = f[n:].tolist()
        while tail and not self.tokenizer.decode(tail[-1:]).strip():  # drop trailing whitespace tokens
            tail.pop()
        return tail

    # ---- stage 1: likelihood scoring -----------------------------------------
    def score_candidates(self, image, prompt, candidates, naive=False, answer_format="{c}"):
        """-> (logprobs {c: sum log p(answer + end-of-turn)}, n_tokens {c: int}); answer = answer_format.format(c=c)."""
        import torch
        tails = {c: self.answer_tokens(prompt, answer_format.format(c=c)) for c in candidates}
        n_tokens = {c: len(t) for c, t in tails.items()}
        prefix = self._encode(self.build_messages(image, prompt), add_generation_prompt=True).to(self.device)
        with torch.no_grad():
            if naive:
                lp = {c: self._score_full(image, prompt, answer_format.format(c=c), prefix["input_ids"][0], tails[c])
                      for c in candidates}
            else:
                lp = self._score_cached(prefix, candidates, tails)
        bad = [c for c, v in lp.items() if not math.isfinite(v)]
        if bad:
            raise FloatingPointError(f"non-finite logprobs for {bad} (dtype={self.dtype}); try bfloat16/float32")
        return lp, n_tokens

    def _score_full(self, image, prompt, answer, prefix_ids, tail):
        """Naive reference (SPEC 5.2): one full forward pass per candidate."""
        ids = self._encode(self.build_messages(image, prompt, answer)).to(self.device)
        x, n, k = ids["input_ids"][0], len(prefix_ids), len(tail)
        assert torch_equal(x[:n], prefix_ids) and x[n:n + k].tolist() == tail, "token alignment mismatch"
        logp = self.model(**ids).logits[0, n - 1:n + k - 1].float().log_softmax(-1)
        return logp.gather(-1, x[n:n + k, None])[:, 0].sum().item()

    def _score_cached(self, prefix, candidates, tails):
        """Run image+prompt once, then score candidates in batches on copies of its KV cache.
        Suffixes are right-padded, so causal attention never lets a real token see padding;
        Qwen3-VL derives M-RoPE positions for the suffix from the cache length + stored rope_deltas."""
        import torch
        out = self.model(**prefix, use_cache=True, logits_to_keep=1)
        first = out.logits[0, -1].float().log_softmax(-1)
        cache = out.past_key_values
        pad = self.tokenizer.pad_token_id or 0
        lp = {}
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            L = max(len(tails[c]) for c in batch)
            ids = torch.full((len(batch), L), pad, dtype=torch.long)
            for j, c in enumerate(batch):
                ids[j, :len(tails[c])] = torch.tensor(tails[c])
            ids = ids.to(self.device)
            kv = copy.deepcopy(cache)
            kv.batch_repeat_interleave(len(batch))
            logp = self.model(input_ids=ids, past_key_values=kv, use_cache=True).logits.float().log_softmax(-1)
            for j, c in enumerate(batch):
                t = tails[c]
                rest = logp[j, :len(t) - 1].gather(-1, ids[j, 1:len(t), None]).sum()
                lp[c] = (first[t[0]] + rest).item()
        return lp

    # ---- stage 2: generation --------------------------------------------------
    def generate(self, image, prompt, max_new_tokens):
        import torch
        ids = self._encode(self.build_messages(image, prompt), add_generation_prompt=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def explain(self, image, country, prompt_version="v1", max_new_tokens=60):
        return clean_rationale(self.generate(image, render_explain(country, prompt_version), max_new_tokens))


def torch_equal(a, b):
    return a.shape == b.shape and bool((a == b).all())


def clean_rationale(text):
    """Strip, collapse whitespace, keep at most two sentences; empty -> fallback.
    No split after single-letter abbreviations ("U.S. roads"); a trailing fragment cut off by
    max_new_tokens is dropped when a complete sentence precedes it."""
    text = " ".join((text or "").split())
    sentences = re.split(r"(?<=[.!?])(?<!\b[A-Z]\.)\s+", text)
    if len(sentences) > 1 and not sentences[-1].endswith((".", "!", "?", '"')):
        sentences.pop()
    text = " ".join(sentences[:2]).strip()
    return text or FALLBACK_RATIONALE


# ---- probabilities / calibration ---------------------------------------------
def softmax_with_temperature(logprobs, T):
    keys = list(logprobs)
    z = np.array([logprobs[k] for k in keys], dtype=float) / T
    p = np.exp(z - z.max())
    return dict(zip(keys, (p / p.sum()).tolist()))


def _nll(L, y, T):
    z = L / T
    z = z - z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))


def fit_temperature(logprob_dicts, labels, lo=0.05, hi=20.0, iters=80):
    """Minimize mean NLL over T in [lo, hi] by golden-section search on log T
    (NLL is convex in 1/T, hence unimodal in log T). -> (T, nll_before(T=1), nll_after)."""
    keys = list(logprob_dicts[0])
    L = np.array([[d[k] for k in keys] for d in logprob_dicts], dtype=float)
    y = np.array([keys.index(l) for l in labels])
    f = lambda u: _nll(L, y, math.exp(u))  # noqa: E731
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    T = math.exp((a + b) / 2)
    return T, _nll(L, y, 1.0), _nll(L, y, T)


def load_temperature(cfg):
    """-> (T, source). outputs/temperature.json (fitted on dev) wins over config."""
    if TEMPERATURE_FILE.exists():
        return float(json.loads(TEMPERATURE_FILE.read_text())["T"]), "temperature.json"
    return float(cfg["inference"].get("temperature", 1.0)), "config"


def apply_temperature(record, T):
    """Recompute probs and top-3 p for a new T. Top-3 order is T-invariant, so rationales stay valid."""
    probs = softmax_with_temperature(record["logprobs"], T)
    record["temperature"] = T
    record["probs"] = {c: round(p, 4) for c, p in probs.items()}
    for t in record["top3"]:
        t["p"] = round(probs[t["country"]], 2)
    return record


# ---- one trial -----------------------------------------------------------------
def candidate_order(candidates, seed, key):
    order = list(candidates)
    random.Random(f"{seed}-{key}").shuffle(order)
    return order


def image_key(data):
    """Ordering key for uploaded images without a trial_id."""
    return hashlib.sha1(data).hexdigest()


def predict_record(gm, image, trial_id, cfg, candidates, temperature, key=None, naive=False):
    """Full SPEC 5.5 record for one image. The single code path shared by predict.py and server.py."""
    inf = cfg["inference"]
    pv = inf["prompt_versions"]
    seed, key = cfg["seed"], key or trial_id
    n_ord = inf.get("n_orderings", 1)
    shuffle = inf.get("shuffle_candidates", True)
    orders = [candidate_order(candidates, seed, key if i == 0 else f"{key}-{i}") if shuffle else list(candidates)
              for i in range(n_ord)]

    t0 = time.time()
    runs = [gm.score_candidates(image, render_rank(o, pv["rank"]), list(candidates),
                                answer_format=RANK_ANSWER[pv["rank"]], naive=naive) for o in orders]
    logprobs = {c: round(sum(r[0][c] for r in runs) / n_ord, 4) for c in candidates}
    t1 = time.time()

    probs = softmax_with_temperature(logprobs, temperature)
    ranked = sorted(candidates, key=lambda c: -logprobs[c])  # stable: ties keep candidate list order
    top3 = [{"country": c, "p": round(probs[c], 2),
             "rationale": gm.explain(image, c, pv["explain"], inf.get("explain_max_new_tokens", 60))}
            for c in ranked[:inf.get("top_k", 3)]]
    t2 = time.time()

    rec = {"trial_id": trial_id, "model_id": gm.model_id, "revision": gm.revision,
           "prompt_versions": dict(pv), "temperature": temperature, "candidate_order": orders[0],
           "logprobs": logprobs, "probs": {c: round(p, 4) for c, p in probs.items()},
           "n_answer_tokens": runs[0][1], "top3": top3, "max_pixels": gm.max_pixels, "seed": seed,
           "runtime_s": round(t2 - t0, 2), "score_s": round(t1 - t0, 2), "explain_s": round(t2 - t1, 2)}
    if n_ord > 1:
        rec["candidate_orders"] = orders
    return rec
