"""Optional LoRA fine-tuning on the OSV-5M train split (SPEC 5.6).

    python src/finetune_lora.py manifest --config config.yaml [--per-country N] [--no-download]
    python src/finetune_lora.py train --config config.yaml [--epochs 1] [--max-steps N] [--out outputs/lora]

manifest: train.csv -> per-country spread-out sample, minus anything within 5 km of a study/dev
image -> data/train/train_manifest.csv (source_id, country, lat, lon, zip, member), images fetched
by remote zip random access into data/train/images/.
train: LoRA on the language model attention projections only (vision tower frozen); the user
turn is the PROMPT_RANK prompt with a seeded candidate shuffle, the loss covers only the scored
answer tokens (RANK_ANSWER + end-of-turn). Saves the adapter, a config snapshot and the manifest hash.
Use the adapter with load_with_adapter(cfg, adapter_dir).
"""
import argparse
import hashlib
import json
import random
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import country_rng, spread_sample  # noqa: E402
from common import DATA, ISO2, OUTPUTS, ROOT, countries, load_config, seed_everything  # noqa: E402
from prompts import RANK_ANSWER, render_rank  # noqa: E402

RAW = DATA / "raw"
TRAIN_DIR = DATA / "train"
MANIFEST = TRAIN_DIR / "train_manifest.csv"
IMG_DIR = TRAIN_DIR / "images"
TRAIN_INDEX = RAW / "train_index.csv.gz"
MANIFEST_COLS = ["source_id", "country", "lat", "lon", "zip", "member"]
EARTH_KM = 6371.0

# LM attention projections only; vision blocks use attn.qkv / attn.proj and never match.
LORA_TARGET = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


# ---------- manifest ----------

def load_train_meta(iso2s, path=None, repo="osv5m/osv5m", chunksize=500_000):
    """Stream train.csv (2.9 GB) in chunks, keeping 4 columns and only the wanted countries."""
    if path is None:
        path = RAW / "train.csv"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo, "train.csv", repo_type="dataset", local_dir=RAW)
    chunks = pd.read_csv(path, usecols=["id", "latitude", "longitude", "country"],
                         dtype={"id": str, "country": str}, keep_default_na=False, chunksize=chunksize)
    return pd.concat([c[c["country"].isin(set(iso2s))] for c in chunks], ignore_index=True)


def reference_points(data_dir=DATA, splits=("study", "dev")):
    """(lat, lon) of every study/dev image, from data/{split}/{split}_meta.csv."""
    frames = [pd.read_csv(p) for s in splits if (p := Path(data_dir) / s / f"{s}_meta.csv").exists()]
    if not frames:
        raise SystemExit(f"no study/dev meta under {data_dir}: run build_dataset.py first "
                         "(train images must be checked against them)")
    m = pd.concat(frames)
    return m["lat"].to_numpy(float), m["lon"].to_numpy(float)


def near_mask(lat, lon, ref_lat, ref_lon, km):
    """True where a point lies within km of any reference point (BallTree, haversine)."""
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(np.c_[ref_lat, ref_lon]), metric="haversine")
    d, _ = tree.query(np.radians(np.c_[lat, lon]), k=1)
    return d[:, 0] * EARTH_KM < km


def sample_manifest(meta, names, per_country, seed, min_km, exclude_km, ref_lat, ref_lon, log):
    """meta: train rows (ISO2 country). -> DataFrame source_id, country (display name), lat, lon."""
    meta = meta[~near_mask(meta["latitude"], meta["longitude"], ref_lat, ref_lon, exclude_km)]
    out = []
    for c in names:
        rows = meta[meta["country"] == ISO2[c]].reset_index(drop=True)
        idx = spread_sample(rows["latitude"], rows["longitude"], per_country, min_km, country_rng(seed, c))
        if len(idx) < per_country:
            log.append(f"SHORT {c}: {len(idx)} < {per_country}")
        r = rows.iloc[idx]
        out.append(pd.DataFrame({"source_id": r["id"], "country": c,
                                 "lat": r["latitude"], "lon": r["longitude"]}))
    return pd.concat(out, ignore_index=True)


def _train_zips(repo):
    from huggingface_hub import HfApi, HfFileSystem
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return HfFileSystem(), sorted(f for f in files if f.startswith("images/train/") and f.endswith(".zip"))


def build_train_index(repo, workers=8):
    """id -> (zip, member) for every train image, read from the remote zip central directories.
    Cached to data/raw/train_index.csv.gz (98 zips, ~3 min first time)."""
    if TRAIN_INDEX.exists():
        return pd.read_csv(TRAIN_INDEX, dtype=str)
    from concurrent.futures import ThreadPoolExecutor
    fs, zips = _train_zips(repo)

    def members(z):
        with fs.open(f"datasets/{repo}/{z}", "rb", block_size=1 << 20) as f, zipfile.ZipFile(f) as zf:
            return [(Path(m).stem, Path(z).name, m) for m in zf.namelist() if m.endswith(".jpg")]

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(members, zips) for r in part]
    idx = pd.DataFrame(rows, columns=["id", "zip", "member"])
    TRAIN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(TRAIN_INDEX, index=False)
    return idx


def download_train_images(manifest, repo, img_dir=IMG_DIR, workers=8):
    """Fetch only the manifest's images out of the remote train zips. Resumable.
    ponytail: same approach as build_dataset.download_images, which is hard-wired to images/test."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfFileSystem
    img_dir.mkdir(parents=True, exist_ok=True)
    todo = manifest[[not (img_dir / f"{s}.jpg").exists() for s in manifest["source_id"]]]
    fs = HfFileSystem()

    def fetch(job):
        z, rows = job
        with fs.open(f"datasets/{repo}/images/train/{z}", "rb", block_size=1 << 18) as f, \
                zipfile.ZipFile(f) as zf:
            for sid, member in rows:
                tmp = img_dir / f"{sid}.jpg.part"
                tmp.write_bytes(zf.read(member))
                tmp.rename(img_dir / f"{sid}.jpg")

    jobs = [(z, list(zip(g["source_id"], g["member"]))[k::workers])
            for z, g in todo.groupby("zip") for k in range(min(workers, len(g)))]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, jobs))
    return len(todo)


def make_manifest(cfg, per_country=None, train_csv=None, data_dir=DATA, out=MANIFEST, download=True):
    d, ft = cfg["data"], cfg.get("finetune", {})
    repo, seed = d.get("hf_repo", "osv5m/osv5m"), cfg["seed"]
    per_country = per_country or ft.get("per_country", 200)
    names = countries(cfg)
    log = []
    meta = load_train_meta([ISO2[c] for c in names], train_csv, repo)
    ref_lat, ref_lon = reference_points(data_dir)
    m = sample_manifest(meta, names, per_country, seed, d.get("min_distance_km", 25),
                        ft.get("exclude_km", 5), ref_lat, ref_lon, log)
    index = build_train_index(repo)
    m = m.merge(index.rename(columns={"id": "source_id"}), on="source_id", how="left")
    missing = m["zip"].isna()
    if missing.any():
        log.append(f"DROP {int(missing.sum())} ids not found in the train zips")
        m = m[~missing]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    m[MANIFEST_COLS].to_csv(out, index=False)
    log.append(f"WROTE {out}: {len(m)} images, per country {m['country'].value_counts().to_dict()}")
    if download:
        log.append(f"downloaded {download_train_images(m, repo)} images -> {IMG_DIR}")
    print("\n".join(log))
    return m


# ---------- training ----------

def lora_config(cfg):
    from peft import LoraConfig
    ft = cfg.get("finetune", {})
    return LoraConfig(r=ft.get("lora_r", 16), lora_alpha=ft.get("lora_alpha", 32),
                      lora_dropout=ft.get("lora_dropout", 0.05), target_modules=LORA_TARGET, bias="none")


def trainable_names(model):
    """Names of trainable params; asserts they are all LoRA weights outside the vision tower."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n in names if "lora_" not in n or "visual" in n]
    assert names and not bad, f"unexpected trainable params: {bad[:5]}"
    return names


def answer_labels(input_ids, n_prefix, tail):
    """Labels for one sequence: -100 everywhere except the answer tokens right after the prefix
    (the model's loss shifts labels internally)."""
    import torch
    k = len(tail)
    assert input_ids[n_prefix:n_prefix + k].tolist() == list(tail), "answer tokens not where expected"
    labels = torch.full_like(input_ids, -100)
    labels[n_prefix:n_prefix + k] = input_ids[n_prefix:n_prefix + k]
    return labels


def encode_example(gm, image, prompt, answer):
    """Processor inputs + labels for one (image, user prompt, assistant answer)."""
    full = gm._encode(gm.build_messages(image, prompt, answer))
    n = gm._encode(gm.build_messages(image, prompt), add_generation_prompt=True)["input_ids"].shape[1]
    full["labels"] = answer_labels(full["input_ids"][0], n, gm.answer_tokens(prompt, answer))[None]
    return full


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_mb():
    """(peak process RSS, accelerator memory) in MB; MPS: driver allocation incl. allocator cache,
    CUDA: peak allocated."""
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)  # bytes
    acc = 0
    if torch.cuda.is_available():
        acc = torch.cuda.max_memory_allocated()
    elif torch.backends.mps.is_available():
        acc = torch.mps.driver_allocated_memory()
    return round(rss / 2**20), round(acc / 2**20)


def train(cfg, manifest=MANIFEST, out=OUTPUTS / "lora", epochs=1, max_steps=None, image_dir=IMG_DIR):
    import torch
    from PIL import Image
    from peft import get_peft_model
    from model import GeoModel, candidate_order

    ft = cfg.get("finetune", {})
    seed, pv = cfg["seed"], cfg["inference"]["prompt_versions"]["rank"]
    accum, lr = ft.get("grad_accum", 8), ft.get("lr", 1e-4)
    seed_everything(seed)
    names = countries(cfg)
    rows = pd.read_csv(manifest, dtype={"source_id": str}).to_dict("records")
    bad = [r["country"] for r in rows if r["country"] not in names]
    assert not bad, f"manifest countries outside the label space: {set(bad)}"

    # ponytail: bf16 base weights; peft keeps the LoRA weights in fp32 (autocast_adapter_dtype).
    gm = GeoModel(cfg, dtype=ft.get("dtype", "bfloat16"))
    model = get_peft_model(gm.model, lora_config(cfg))
    gm.model = model
    tn = trainable_names(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {len(tn)} tensors, {n_train:,} params, e.g. {tn[0]}")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    stream = []
    for e in range(epochs):
        order = list(rows)
        random.Random(f"{seed}-epoch{e}").shuffle(order)
        stream += [(e, r) for r in order]
    if max_steps:
        stream = stream[:max_steps * accum]

    step, window, log, t0 = 0, [], [], time.time()
    for i, (e, r) in enumerate(stream):
        path = Path(r["path"]) if r.get("path") else Path(image_dir) / f"{r['source_id']}.jpg"
        image = Image.open(path).convert("RGB")
        prompt = render_rank(candidate_order(names, seed, f"train-{r['source_id']}-e{e}"), pv)
        batch = encode_example(gm, image, prompt, RANK_ANSWER[pv].format(c=r["country"])).to(gm.device)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss on {r['source_id']}")
        # ponytail: a final partial window is still divided by accum (slightly smaller last step).
        (loss / accum).backward()
        window.append(loss.item())
        if len(window) == accum or i == len(stream) - 1:
            opt.step()
            opt.zero_grad()
            step += 1
            rss, acc = memory_mb()
            dt = time.time() - t0
            log.append({"step": step, "epoch": e, "loss": float(np.mean(window)), "s": round(dt, 1),
                        "peak_rss_mb": rss, "accel_mb": acc})
            print(f"step {step} epoch {e} loss {log[-1]['loss']:.4f} {dt:.1f}s rss {rss} MB accel {acc} MB",
                  flush=True)
            window, t0 = [], time.time()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "train_info.json").write_text(json.dumps({
        "model_id": gm.model_id, "revision": gm.revision, "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest), "n_images": len(rows), "epochs": epochs,
        "max_steps": max_steps, "grad_accum": accum, "lr": lr, "dtype": str(gm.dtype),
        "device": gm.device, "prompt_version_rank": pv, "trainable_params": n_train, "log": log}, indent=2))
    print(f"saved adapter -> {out}")
    return log


def load_with_adapter(cfg, adapter_dir):
    """GeoModel with a trained LoRA adapter merged in, so every scoring path works unchanged."""
    from peft import PeftModel
    from model import GeoModel
    gm = GeoModel(cfg)
    gm.model = PeftModel.from_pretrained(gm.model, str(adapter_dir)).merge_and_unload().eval()
    gm.adapter = str(adapter_dir)
    return gm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--config", default=str(ROOT / "config.yaml"))
    m.add_argument("--per-country", type=int, help="default: finetune.per_country")
    m.add_argument("--train-csv", help="local OSV-5M train.csv (default: data/raw/train.csv, downloaded)")
    m.add_argument("--no-download", action="store_true", help="write the manifest only")
    t = sub.add_parser("train")
    t.add_argument("--config", default=str(ROOT / "config.yaml"))
    t.add_argument("--manifest", default=str(MANIFEST), help="CSV with source_id,country[,path]")
    t.add_argument("--image-dir", default=str(IMG_DIR))
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--max-steps", type=int, help="stop after N optimizer steps")
    t.add_argument("--out", default=str(OUTPUTS / "lora"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "manifest":
        make_manifest(cfg, a.per_country, a.train_csv, download=not a.no_download)
    else:
        train(cfg, a.manifest, a.out, a.epochs, a.max_steps, a.image_dir)


if __name__ == "__main__":
    main()
