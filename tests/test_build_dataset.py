import zlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageEnhance

import build_dataset as bd


def noise_image(seed, size=(320, 300)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1] // 20, size[0] // 20, 3), dtype=np.uint8)).resize(size)


def phash_int(im):
    import imagehash
    return int(str(imagehash.phash(im)), 16)


def test_haversine_known_distance():
    assert bd.haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.5, abs=1.5)
    assert bd.haversine_km(10, 20, 10, 20) == 0


def test_spread_sample_honors_min_distance_and_is_seeded():
    rng = np.random.default_rng(0)
    # 50 points within ~1 km of each other + 10 points on a 1-degree grid (~111 km apart)
    lat = np.r_[40 + rng.uniform(0, 0.01, 50), 30 + np.arange(10)]
    lon = np.r_[-100 + rng.uniform(0, 0.01, 50), np.full(10, -90.0)]
    idx = bd.spread_sample(lat, lon, 11, 25, np.random.default_rng(1))
    assert len(set(idx)) == 11
    assert bd.min_dist_km(lat[idx], lon[idx]).min() >= 25  # 10 grid points + one cluster point
    assert idx == bd.spread_sample(lat, lon, 11, 25, np.random.default_rng(1))


def test_spread_sample_tops_up_when_too_dense():
    lat, lon = 40 + np.arange(5) * 0.001, np.zeros(5)  # all within 1 km
    idx = bd.spread_sample(lat, lon, 4, 25, np.random.default_rng(0))
    assert len(set(idx)) == 4
    assert len(bd.spread_sample(lat, lon, 10, 25, np.random.default_rng(0))) == 5


def test_dedup_phash_drops_near_duplicates():
    a, b = noise_image(1), noise_image(2)
    a2 = ImageEnhance.Brightness(a).enhance(1.1).resize((330, 310))
    items = [("a", phash_int(a)), ("b", phash_int(b)), ("a2", phash_int(a2))]
    kept, dropped = bd.dedup_phash(items, 6)
    assert kept == ["a", "b"] and dropped == ["a2"]


def test_image_quality(tmp_path):
    noise_image(3, (400, 200)).save(tmp_path / "x.jpg")
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    short, h = bd.image_quality(tmp_path / "x.jpg")
    assert short == 200 and isinstance(h, int)
    assert bd.image_quality(tmp_path / "bad.jpg") is None


@pytest.mark.parametrize("text,script", [
    ("STOP Main St", "Latin"), ("Москва 5", "Cyrillic"), ("서울역", "Hangul"),
    ("東京タワー", "Japanese"), ("北京", "Han"), ("กรุงเทพ", "Thai"),
    ("दिल्ली", "Devanagari"), ("Ελλάδα", "other"), ("123 !!", ""), ("", ""),
    ("Seoul 서울특별시청", "Hangul"),
])
def test_detect_script(text, script):
    assert bd.detect_script(text) == script


def test_tag_text_thresholds():
    dets = [("AB", 0.99), ("...", 0.9), ("Lowconf", 0.3), (" Main St ", 0.8)]
    assert bd.tag_text(dets, 0.5, 3) == (True, "Main St", "Latin")
    assert bd.tag_text(dets[:3], 0.5, 3) == (False, "", "")


def test_resolve_countries_substitutes_backup():
    counts = {"US": 500, "IE": 10, "NZ": 50, "NL": 400, "ID": 900}
    log = []
    active = bd.resolve_countries(counts, ["United States", "Ireland"], ["New Zealand", "Netherlands"], 300, log)
    assert active == ["United States", "Netherlands"]
    assert log and "SUBSTITUTE Ireland" in log[0]
    log = []
    assert bd.resolve_countries(counts, ["Ireland"], ["New Zealand"], 300, log) == []
    assert "DROP Ireland" in log[0]


QUOTAS = [("study", 5, 5), ("dev", 5, 5)]


def cands(n_text, n_notext):
    c = [(f"t{i}", True) for i in range(n_text)] + [(f"n{i}", False) for i in range(n_notext)]
    return sorted(c, key=lambda x: zlib.crc32(x[0].encode()))  # interleaved, deterministic


def test_assign_splits_stratified_and_disjoint():
    log = []
    out = bd.assign_splits(cands(15, 15), QUOTAS, log)
    s, d = out["study"], out["dev"]
    assert len(s) == len(d) == 10 and not set(s) & set(d)
    assert sum(x.startswith("t") for x in s) == 5 and sum(x.startswith("t") for x in d) == 5
    assert log == []


def test_assign_splits_fills_from_other_stratum_and_logs():
    log = []
    out = bd.assign_splits(cands(7, 20), QUOTAS, log, "Chile")
    assert sum(x.startswith("t") for x in out["study"]) == 5
    assert sum(x.startswith("t") for x in out["dev"]) == 2 and len(out["dev"]) == 10
    assert any(l.startswith("FILL Chile dev: 3 text short, filled 3") for l in log)
    log = []
    out = bd.assign_splits(cands(2, 3), QUOTAS, log, "Chile")
    assert len(out["study"]) == 5 and out["dev"] == []
    assert any("SHORTFALL Chile study" in l for l in log)


def test_assign_trial_ids_is_seeded_permutation():
    ids = bd.assign_trial_ids(200, "S", 42)
    assert sorted(ids) == [f"S{i:03d}" for i in range(1, 201)]
    assert ids == bd.assign_trial_ids(200, "S", 42)
    assert ids != sorted(ids) and ids != bd.assign_trial_ids(200, "S", 43)
    assert bd.assign_trial_ids(3, "D", 42)[0].startswith("D")


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """Offline build: synthetic metadata, images and OCR; all paths under tmp_path."""
    data = tmp_path / "data"
    raw = data / "raw"
    for name, val in {"ROOT": tmp_path, "DATA": data, "RAW": raw, "IMG_DIR": raw / "images",
                      "INDEX_CSV": raw / "test_index.csv", "OCR_CACHE": raw / "ocr_cache.csv",
                      "LOG_FILE": data / "build_log.txt"}.items():
        monkeypatch.setattr(bd, name, val)
    rng = np.random.default_rng(0)
    rows = []
    for iso, lat0 in (("CL", -33), ("JP", 35), ("KR", 37)):
        for i in range(40):
            rows.append((f"{iso}{i:03d}", lat0 + rng.uniform(-3, 3), 100 + rng.uniform(-3, 3), iso))
    meta = pd.DataFrame(rows, columns=["id", "latitude", "longitude", "country"])
    monkeypatch.setattr(bd, "load_meta", lambda repo: meta)
    monkeypatch.setattr(bd, "build_index", lambda repo: pd.DataFrame({"id": meta["id"], "zip": "00.zip", "member": ""}))

    def fake_download(ids, index, repo, img_dir=None):
        bd.IMG_DIR.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(sorted(ids)):
            noise_image(k, (300, 200 if i.endswith("7") else 300)).save(bd.IMG_DIR / f"{i}.jpg")
        return len(ids)
    monkeypatch.setattr(bd, "download_images", fake_download)
    monkeypatch.setattr(bd, "run_ocr", lambda jobs, gpu: {
        j[0]: [("SIGN TEXT", 0.9)] if int(j[0][2:]) % 3 == 0 else [] for j in jobs})
    cfg = bd.load_config()
    cfg["data"].update(countries=["Chile", "Japan", "South Korea"], backup_countries=[], min_candidates=30,
                       per_country_study=4, per_country_dev=4, text_ratio=0.5)
    return cfg, data


def test_build_end_to_end_and_deterministic(fake_repo):
    cfg, data = fake_repo
    bd.build(cfg, contact_sheet=True, pool=20)
    s1 = pd.read_csv(data / "study" / "study_meta.csv", dtype={"source_id": str})
    d1 = pd.read_csv(data / "dev" / "dev_meta.csv", dtype={"source_id": str})
    assert list(s1.columns) == bd.META_COLS
    assert len(s1) == len(d1) == 12 and not set(s1.source_id) & set(d1.source_id)
    assert set(s1.trial_id) == {f"S{i:03d}" for i in range(1, 13)}
    assert (data / "study" / "S001.jpg").exists() and (data / "dev" / "contact_sheet.html").exists()
    assert s1.groupby("country").size().to_dict() == {"Chile": 4, "Japan": 4, "South Korea": 4}
    assert set(s1.cluster) == {"South America", "East Asia"}
    assert not s1.source_id.str.endswith("7").any()  # short side < 256 filtered
    assert s1.groupby("country").has_text.sum().eq(2).all()
    log = (data / "build_log.txt").read_text()
    assert "FILTER" in log and "WROTE study: 12" in log

    bd.build(cfg, pool=20)  # deterministic rebuild
    assert pd.read_csv(data / "study" / "study_meta.csv", dtype={"source_id": str}).equals(s1)


def test_build_unstratified(fake_repo):
    cfg, data = fake_repo
    cfg["data"]["text_ratio"] = None
    bd.build(cfg, pool=20)
    s1 = pd.read_csv(data / "study" / "study_meta.csv", dtype={"source_id": str})
    assert len(s1) == 12 and s1.groupby("country").size().eq(4).all()
    assert "FILL" not in (data / "build_log.txt").read_text()


def test_build_writes_countries_txt_on_substitution(fake_repo):
    cfg, data = fake_repo
    cfg["data"].update(countries=["Chile", "Ireland"], backup_countries=["Japan"])
    bd.build(cfg, splits=["study"], pool=20)
    assert (data / "countries.txt").read_text().split("\n")[:2] == ["Chile", "Japan"]
    assert "SUBSTITUTE Ireland" in (data / "build_log.txt").read_text()


@pytest.mark.network
def test_download_one_real_image(tmp_path):
    import zipfile
    fs, zips = bd.zip_paths("osv5m/osv5m")
    assert len(zips) == 5
    with fs.open(zips[-1], "rb", block_size=1 << 20) as f, zipfile.ZipFile(f) as zf:
        member = next(m for m in zf.namelist() if m.endswith(".jpg"))
    sid = member.split("/")[-1][:-4]
    index = pd.DataFrame({"id": [sid], "zip": [zips[-1].split("/")[-1]], "member": [member]})
    assert bd.download_images([sid], index, "osv5m/osv5m", tmp_path) == 1
    assert bd.image_quality(tmp_path / f"{sid}.jpg")[0] >= 256
    assert bd.download_images([sid], index, "osv5m/osv5m", tmp_path) == 0  # cached


def test_coord_overlay_regex():
    assert bd.COORD_RE.search("114km/h N49.6952 Е40.5379 | AdvoCam")
    assert bd.COORD_RE.search("n12 52.7158")
    assert not bd.COORD_RE.search("2016/11/04 08.39.51 | Main St 12.5")
