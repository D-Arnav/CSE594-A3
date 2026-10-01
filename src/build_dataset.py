"""Build the study and dev image sets from the OSV-5M test split (SPEC section 4).

Pipeline: metadata -> country check/backups -> per-country spread-out candidate pool ->
download (cached) -> quality filter + phash dedup -> EasyOCR text tagging (cached) ->
(optionally text-stratified) disjoint study/dev sampling ->
data/{split}/{S,D}NNN.jpg + {split}_meta.csv, data/build_log.txt.

    python src/build_dataset.py --config config.yaml [--splits study dev] [--contact-sheet]
                                [--limit-countries N] [--pool-per-country N]
"""
import argparse
import html
import json
import re
import shutil
import sys
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLUSTER_OF, DATA, ISO2, ROOT, load_config  # noqa: E402

RAW = DATA / "raw"
IMG_DIR = RAW / "images"
INDEX_CSV = RAW / "test_index.csv"
OCR_CACHE = RAW / "ocr_cache.csv"
LOG_FILE = DATA / "build_log.txt"
SPLIT_PREFIX = {"study": "S", "dev": "D"}
META_COLS = ["trial_id", "image_path", "source_id", "country", "cluster", "lat", "lon",
             "has_text", "ocr_text", "script", "split"]

# ponytail: one EasyOCR reader per country (its main script + English), not every script on
# every image. Text in another script (e.g. Han signs in Korea) may be missed or garbled.
OCR_LANGS = {"Russia": ["ru", "en"], "South Korea": ["ko", "en"], "Japan": ["ja", "en"],
             "Thailand": ["th", "en"], "India": ["hi", "en"]}

# Decimal GPS coordinates burnt in by dashcams (e.g. "N49.6952 E40.5379"): a location leak.
COORD_RE = re.compile(r"\d{1,3}[.,]\d{4,}")

SCRIPT_RANGES = [  # (name, [(lo, hi), ...]) by Unicode code point
    ("Latin", [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)]),
    ("Cyrillic", [(0x400, 0x52F)]),
    ("Hangul", [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
    ("Japanese", [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)]),  # kana
    ("Han", [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
    ("Thai", [(0xE00, 0xE7F)]),
    ("Devanagari", [(0x900, 0x97F)]),
]


# ---------- pure helpers ----------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; numpy-broadcasting."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spread_sample(lat, lon, k, min_km, rng):
    """Pick k indices, preferring points >= min_km apart.

    Greedy pass over a seeded random order accepts a point only if it is >= min_km from all
    accepted ones; if that yields < k, top up by farthest-point sampling over the rest.
    ponytail: random-order greedy instead of pure farthest-point first, because pure FPS
    picks only the remote periphery of a country; FPS is the fallback.
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n = len(lat)
    k = min(k, n)
    chosen = []
    mind = np.full(n, np.inf)  # distance from each point to the nearest chosen point
    for i in rng.permutation(n):
        if len(chosen) == k:
            break
        if mind[i] >= min_km:
            chosen.append(int(i))
            mind = np.minimum(mind, haversine_km(lat[i], lon[i], lat, lon))
    while len(chosen) < k:
        d = mind.copy()
        d[chosen] = -1
        i = int(np.argmax(d))
        chosen.append(i)
        mind = np.minimum(mind, haversine_km(lat[i], lon[i], lat, lon))
    return chosen


def min_dist_km(lat, lon):
    """For each point, distance to its nearest other point."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def dedup_phash(items, max_dist):
    """items: [(source_id, int_hash)] in priority order. Keep the first of every near-duplicate group."""
    kept, kept_h, dropped = [], [], []
    for sid, h in items:
        if any((h ^ g).bit_count() <= max_dist for g in kept_h):
            dropped.append(sid)
        else:
            kept.append(sid)
            kept_h.append(h)
    return kept, dropped


def detect_script(text):
    """Dominant script of text; any kana -> Japanese. '' if no letters."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        name = next((s for s, rs in SCRIPT_RANGES if any(lo <= cp <= hi for lo, hi in rs)), None)
        if name is None:
            if not ch.isalpha():
                continue
            name = "other"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    if "Japanese" in counts:
        return "Japanese"
    return max(counts, key=counts.get)


def tag_text(detections, ocr_conf, min_chars):
    """detections: [(text, conf)]. Returns (has_text, ocr_text, script)."""
    good = [t.strip() for t, c in detections
            if c >= ocr_conf and sum(ch.isalnum() for ch in t) >= min_chars]
    text = " | ".join(good)
    return bool(good), text, detect_script(text)


def resolve_countries(counts, primary, backups, min_n, log):
    """Replace any primary country with < min_n source rows by the next backup that has enough."""
    backups = [b for b in backups if counts.get(ISO2[b], 0) >= min_n]
    active = []
    for c in primary:
        n = counts.get(ISO2[c], 0)
        if n >= min_n:
            active.append(c)
        elif backups:
            b = backups.pop(0)
            log.append(f"SUBSTITUTE {c} ({n} < {min_n} candidates) -> {b} ({counts[ISO2[b]]})")
            active.append(b)
        else:
            log.append(f"DROP {c}: {n} < {min_n} candidates and no backup left")
    return active


def assign_splits(cands, quotas, log, country=""):
    """Stratified, disjoint assignment of one country's candidates.

    cands: [(source_id, has_text)] in seeded order. quotas: [(split, n_text, n_notext)] in fill
    order. A short stratum is filled from the other one and logged.
    Returns {split: [source_id, ...]}.
    """
    pools = {True: [s for s, t in cands if t], False: [s for s, t in cands if not t]}
    out = {}
    for split, n_text, n_notext in quotas:
        picks = []
        for want, n in ((True, n_text), (False, n_notext)):
            take = pools[want][:n]
            pools[want] = pools[want][n:]
            if len(take) < n:
                extra = pools[not want][:n - len(take)]
                pools[not want] = pools[not want][len(extra):]
                log.append(f"FILL {country} {split}: {n - len(take)} {'text' if want else 'no-text'} "
                           f"short, filled {len(extra)} from other stratum")
                take += extra
                if len(take) < n:
                    log.append(f"SHORTFALL {country} {split}: missing {n - len(take)}")
            picks += take
        out[split] = picks
    return out


def assign_trial_ids(n, prefix, seed):
    """Seeded shuffled IDs: slot i gets ids[i]. Slots are built in country order, so the
    shuffle keeps IDs from leaking country."""
    order = np.random.default_rng([seed, zlib.crc32(prefix.encode())]).permutation(n)
    return [f"{prefix}{j + 1:03d}" for j in order]


def country_rng(seed, country):
    return np.random.default_rng([seed, zlib.crc32(country.encode())])


# ---------- I/O stages ----------

def load_meta(repo):
    path = RAW / "test.csv"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, "test.csv", repo_type="dataset", local_dir=RAW)
    return pd.read_csv(path, usecols=["id", "latitude", "longitude", "country"],
                       dtype={"id": str, "country": str}, keep_default_na=False)


def zip_paths(repo):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    return fs, sorted(fs.glob(f"datasets/{repo}/images/test/*.zip"))


def build_index(repo):
    """id -> (zip, member) for every test image; cached to data/raw/test_index.csv."""
    if INDEX_CSV.exists():
        return pd.read_csv(INDEX_CSV, dtype=str)
    fs, zips = zip_paths(repo)
    rows = []
    for z in zips:
        with fs.open(z, "rb", block_size=1 << 20) as f, zipfile.ZipFile(f) as zf:
            rows += [(Path(m).stem, Path(z).name, m) for m in zf.namelist() if m.endswith(".jpg")]
    idx = pd.DataFrame(rows, columns=["id", "zip", "member"])
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(INDEX_CSV, index=False)
    return idx


def download_images(ids, index, repo, img_dir=IMG_DIR, workers=8):
    """Fetch only the listed images out of the remote zips (random access, parallel). Resumable."""
    from concurrent.futures import ThreadPoolExecutor
    img_dir.mkdir(parents=True, exist_ok=True)
    todo = index[index["id"].isin([i for i in ids if not (img_dir / f"{i}.jpg").exists()])]
    if todo.empty:
        return 0
    fs, _ = zip_paths(repo)

    def fetch(job):  # one zip handle per job; jobs split each zip's members across workers
        z, rows = job
        with fs.open(f"datasets/{repo}/images/test/{z}", "rb", block_size=1 << 18) as f, \
                zipfile.ZipFile(f) as zf:
            for sid, member in rows:
                tmp = img_dir / f"{sid}.jpg.part"
                tmp.write_bytes(zf.read(member))
                tmp.rename(img_dir / f"{sid}.jpg")

    jobs = [(z, list(zip(g["id"], g["member"]))[k::workers])
            for z, g in todo.groupby("zip") for k in range(min(workers, len(g)))]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, jobs))
    return len(todo)


def image_quality(path):
    """(short_side, int phash) or None if unreadable."""
    import imagehash
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
            return min(im.size), int(str(imagehash.phash(im)), 16)
    except Exception:
        return None


def run_ocr(jobs, gpu):
    """jobs: [(source_id, path, langs)]. Returns {source_id: [(text, conf)]}, cached on disk."""
    cache = {}
    if OCR_CACHE.exists():
        c = pd.read_csv(OCR_CACHE, dtype=str, keep_default_na=False)
        cache = {r.source_id: [tuple(d) for d in json.loads(r.detections)] for r in c.itertuples()}
    todo = [j for j in jobs if j[0] not in cache]
    readers = {}
    new_file = not OCR_CACHE.exists()
    with open(OCR_CACHE, "a", newline="") as f:
        if new_file:
            f.write("source_id,langs,detections\n")
        for sid, path, langs in todo:
            key = tuple(langs)
            if key not in readers:
                import easyocr
                readers[key] = easyocr.Reader(langs, gpu=gpu, verbose=False)
            dets = [(t, round(float(c), 4)) for _, t, c in readers[key].readtext(str(path))]
            cache[sid] = dets
            pd.DataFrame([[sid, "+".join(langs), json.dumps(dets, ensure_ascii=False)]]).to_csv(
                f, header=False, index=False)
            f.flush()
    return {j[0]: cache[j[0]] for j in jobs}


def write_contact_sheet(meta, split_dir):
    cells = "\n".join(
        f'<figure><img src="{Path(r.image_path).name}" loading="lazy"><figcaption><b>{r.trial_id}</b> '
        f'{html.escape(r.country)} &middot; {r.source_id}<br>{html.escape(str(r.ocr_text))}</figcaption></figure>'
        for r in meta.itertuples())
    (split_dir / "contact_sheet.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Contact sheet</title><style>"
        "body{font:13px sans-serif;display:flex;flex-wrap:wrap;gap:8px}"
        "figure{margin:0;width:340px}img{width:100%}</style>\n" + cells)


# ---------- main ----------

def build(cfg, splits=("study", "dev"), contact_sheet=False, limit_countries=None, pool=None):
    d = cfg["data"]
    seed = cfg["seed"]
    repo = d.get("hf_repo", "osv5m/osv5m")
    pool = pool or d.get("pool_per_country", 60)
    log = [f"build {datetime.now().isoformat(timespec='seconds')} seed={seed} pool={pool}"]

    DATA.mkdir(parents=True, exist_ok=True)
    meta = load_meta(repo)
    counts = meta["country"].value_counts().to_dict()
    primary = list(d["countries"])
    active = resolve_countries(counts, primary, d.get("backup_countries", []), d["min_candidates"], log)
    if active != primary:
        (DATA / "countries.txt").write_text("\n".join(active) + "\n")
        log.append("wrote data/countries.txt")
    if limit_countries:
        active = active[:limit_countries]
        log.append(f"limit-countries: {active}")

    index = build_index(repo)
    meta = meta[meta["id"].isin(set(index["id"]))]

    # 1. spread-out candidate pool per country
    pools = {}
    for c in active:
        rows = meta[meta["country"] == ISO2[c]].reset_index(drop=True)
        idx = spread_sample(rows["latitude"], rows["longitude"], pool, d["min_distance_km"],
                            country_rng(seed, c))
        pools[c] = rows.iloc[idx]
        near = sum(min_dist_km(pools[c]["latitude"], pools[c]["longitude"]) < d["min_distance_km"])
        log.append(f"POOL {c}: {len(idx)} of {len(rows)}, {near} closer than {d['min_distance_km']} km to another")
    all_ids = [i for c in active for i in pools[c]["id"]]

    t = time.time()
    n_dl = download_images(all_ids, index, repo)
    if n_dl:
        log.append(f"downloaded {n_dl} images, {(time.time() - t) / n_dl:.2f} s/img")

    # 2. quality filter + phash dedup across all candidates
    q = {i: image_quality(IMG_DIR / f"{i}.jpg") for i in all_ids}
    bad = [i for i in all_ids if q[i] is None]
    small = [i for i in all_ids if q[i] and q[i][0] < d["min_short_side"]]
    ok, dups = dedup_phash([(i, q[i][1]) for i in all_ids if q[i] and q[i][0] >= d["min_short_side"]],
                           d["phash_max_dist"])
    log.append(f"FILTER unreadable={len(bad)} small={len(small)} near-dup={len(dups)} kept={len(ok)}")
    ok = set(ok)

    # 3. OCR text tagging
    t = time.time()
    jobs = [(i, IMG_DIR / f"{i}.jpg", OCR_LANGS.get(c, ["en"])) for c in active
            for i in pools[c]["id"] if i in ok]
    dets = run_ocr(jobs, gpu=d.get("ocr_gpu", True))
    log.append(f"OCR {len(jobs)} images in {time.time() - t:.1f} s (cached ones are free)")
    tags = {i: tag_text(dets[i], d["ocr_conf"], d["ocr_min_chars"]) for i, _, _ in jobs}
    overlay = {i for i, (_, text, _) in tags.items() if COORD_RE.search(text)}
    ok -= overlay
    log.append(f"FILTER gps-overlay={len(overlay)} kept={len(ok)}")

    # 4. stratified disjoint sampling
    # ponytail: text_ratio null = no text stratification. EasyOCR at OSV-5M resolution finds
    # ~0 readable text in most countries, so 5+5 per country is not reachable.
    ratio = d.get("text_ratio")
    quotas = []
    for s in ("study", "dev"):  # always both, so either split is stable on its own
        n = d[f"per_country_{s}"]
        nt = round(n * ratio) if ratio is not None else 0
        quotas.append((s, nt, n - nt))
    slots = {s: [] for s, _, _ in quotas}
    for c in active:
        cands = [(i, tags[i][0] if ratio is not None else False) for i in pools[c]["id"] if i in ok]
        nt = sum(tags[i][0] for i, _ in cands)
        log.append(f"TEXT {c}: {nt} OCR-text / {len(cands) - nt} no-text candidates")
        picks = assign_splits(cands, quotas, log, c)
        for s in picks:
            slots[s] += [(c, sid) for sid in picks[s]]

    lat = {r.id: (r.latitude, r.longitude) for c in active for r in pools[c].itertuples()}

    def row(c, sid):
        has, text, script = tags[sid]
        return {"source_id": sid, "country": c, "cluster": CLUSTER_OF[c], "lat": lat[sid][0],
                "lon": lat[sid][1], "has_text": has, "ocr_text": text, "script": script}

    # 5. write images + meta
    for s in splits:
        prefix = SPLIT_PREFIX[s]
        ids = assign_trial_ids(len(slots[s]), prefix, seed)
        out_dir = DATA / s
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{prefix}[0-9][0-9][0-9].jpg"):
            old.unlink()
        rows = []
        for tid, (c, sid) in zip(ids, slots[s]):
            dst = out_dir / f"{tid}.jpg"
            shutil.copyfile(IMG_DIR / f"{sid}.jpg", dst)
            rows.append({"trial_id": tid, "image_path": dst.relative_to(ROOT).as_posix(),
                         **row(c, sid), "split": s})
        m = pd.DataFrame(rows, columns=META_COLS).sort_values("trial_id")
        m.to_csv(out_dir / f"{s}_meta.csv", index=False)
        log.append(f"WROTE {s}: {len(m)} images, has_text={int(m['has_text'].sum())}")
        if contact_sheet:
            write_contact_sheet(m, out_dir)

    LOG_FILE.write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--splits", nargs="+", default=["study", "dev"], choices=list(SPLIT_PREFIX))
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument("--limit-countries", type=int, help="first N countries only (smoke test)")
    ap.add_argument("--pool-per-country", type=int)
    a = ap.parse_args()
    build(load_config(a.config), a.splits, a.contact_sheet, a.limit_countries, a.pool_per_country)


if __name__ == "__main__":
    main()
