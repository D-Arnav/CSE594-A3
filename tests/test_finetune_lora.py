import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import finetune_lora as fl
from build_dataset import haversine_km

KM_PER_DEG = 111.195  # 1 degree of latitude


def test_near_mask_radius():
    ref_lat, ref_lon = np.array([40.0, -30.0]), np.array([-100.0, 140.0])
    lat = np.array([40.0 + 4 / KM_PER_DEG, 40.0 + 6 / KM_PER_DEG, -30.0, 10.0])
    lon = np.array([-100.0, -100.0, 140.0 + 0.01, 10.0])
    assert fl.near_mask(lat, lon, ref_lat, ref_lon, 5).tolist() == [True, False, True, False]


def synthetic_meta(rng):
    """Japan: 300 rows, France: 40 rows (short), plus 50 Japan rows clustered at a 'study' point."""
    rows = []
    for iso, n, (la, lo) in (("JP", 300, (36, 138)), ("FR", 40, (46, 2))):
        rows += [(f"{iso}{i}", la + rng.uniform(-3, 3), lo + rng.uniform(-3, 3), iso) for i in range(n)]
    rows += [(f"JPnear{i}", 35.0 + rng.uniform(-0.02, 0.02), 139.0, "JP") for i in range(50)]
    rows += [(f"US{i}", 40.0, -100.0, "US") for i in range(5)]  # not in the label space
    return pd.DataFrame(rows, columns=["id", "latitude", "longitude", "country"])


def test_sample_manifest_balance_and_exclusion():
    meta = synthetic_meta(np.random.default_rng(0))
    ref_lat, ref_lon = np.array([35.0]), np.array([139.0])
    log = []
    m = fl.sample_manifest(meta, ["Japan", "France"], 100, 42, 10, 5, ref_lat, ref_lon, log)
    counts = m["country"].value_counts().to_dict()
    assert counts == {"Japan": 100, "France": 40}
    assert log == ["SHORT France: 40 < 100"]
    assert m["source_id"].is_unique and not m["source_id"].str.startswith("JPnear").any()
    assert haversine_km(m["lat"].to_numpy(), m["lon"].to_numpy(), 35.0, 139.0).min() >= 5
    again = fl.sample_manifest(meta, ["Japan", "France"], 100, 42, 10, 5, ref_lat, ref_lon, [])
    assert again.equals(m)  # seeded
    other = fl.sample_manifest(meta, ["Japan", "France"], 100, 7, 10, 5, ref_lat, ref_lon, [])
    assert set(other["source_id"]) != set(m["source_id"])


def test_load_train_meta_streams_and_filters(tmp_path):
    p = tmp_path / "train.csv"
    pd.DataFrame({"id": ["001", "2", "3", "4", "5"], "latitude": [1, 2, 3, 4, 5],
                  "longitude": [1, 2, 3, 4, 5], "country": ["JP", "FR", "US", "JP", "NA"],
                  "city": list("abcde")}).to_csv(p, index=False)
    m = fl.load_train_meta(["JP", "FR"], p, chunksize=2)
    assert list(m.columns) == ["id", "latitude", "longitude", "country"]
    assert m["id"].tolist() == ["001", "2", "4"]  # ids stay strings, "NA" (Namibia) is not NaN


def test_make_manifest_end_to_end(tmp_path, monkeypatch):
    meta = synthetic_meta(np.random.default_rng(1))
    meta.to_csv(tmp_path / "train.csv", index=False)
    for split in ("study", "dev"):
        (tmp_path / split).mkdir()
        pd.DataFrame({"lat": [35.0], "lon": [139.0]}).to_csv(tmp_path / split / f"{split}_meta.csv", index=False)
    index = pd.DataFrame({"id": meta["id"], "zip": "00.zip", "member": "00/" + meta["id"] + ".jpg"})
    monkeypatch.setattr(fl, "build_train_index", lambda repo: index[index["id"] != "FR0"])
    cfg = {"seed": 42, "data": {"countries": ["Japan", "France"], "min_distance_km": 10},
           "finetune": {"per_country": 50}}
    monkeypatch.setattr(fl, "countries", lambda cfg: cfg["data"]["countries"])
    out = tmp_path / "train" / "train_manifest.csv"
    fl.make_manifest(cfg, None, tmp_path / "train.csv", tmp_path, out, download=False)
    m = pd.read_csv(out, dtype=str)
    assert list(m.columns) == fl.MANIFEST_COLS
    assert m["country"].value_counts().to_dict() == {"Japan": 50, "France": 39}  # FR0 not in any zip
    assert (m["member"] == "00/" + m["source_id"] + ".jpg").all()
    assert not m["source_id"].str.startswith("JPnear").any()


def test_reference_points_requires_meta(tmp_path):
    with pytest.raises(SystemExit):
        fl.reference_points(tmp_path)


def test_answer_labels_masks_everything_but_answer():
    ids = torch.tensor([5, 6, 7, 8, 42, 43, 99, 10])
    labels = fl.answer_labels(ids, 4, [42, 43, 99])
    assert labels.tolist() == [-100, -100, -100, -100, 42, 43, 99, -100]
    with pytest.raises(AssertionError):
        fl.answer_labels(ids, 3, [42, 43, 99])


def test_encode_example_with_real_processor():
    """Tokenizer/processor only (no weights): the labelled tokens decode to the answer + <|im_end|>."""
    from PIL import Image
    from model import GeoModel
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                             revision="89644892e4d85e24eaac8bacfd4f463576704203")
    except Exception as e:  # not in the HF cache
        pytest.skip(f"processor unavailable: {e}")
    gm = GeoModel.__new__(GeoModel)
    gm.processor, gm.tokenizer = proc, proc.tokenizer
    image = Image.new("RGB", (64, 64), "gray")
    batch = fl.encode_example(gm, image, "Which country? Answer with the country name only.", "South Korea")
    labels = batch["labels"][0]
    kept = labels[labels != -100]
    assert proc.tokenizer.decode(kept) == "South Korea<|im_end|>"
    assert "pixel_values" in batch and labels.shape == batch["input_ids"][0].shape
    first = int((labels != -100).nonzero()[0])
    assert proc.tokenizer.decode(batch["input_ids"][0, :first]).endswith("<|im_start|>assistant\n")


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, nn.Linear(4, 4))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attn()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 4)


class Tiny(nn.Module):
    """Mimics Qwen3-VL module names, including a q_proj-named trap inside the vision tower."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([nn.Module()])
        self.model.visual.blocks[0].attn = nn.Module()
        self.model.visual.blocks[0].attn.qkv = nn.Linear(4, 12)
        self.model.visual.blocks[0].attn.q_proj = nn.Linear(4, 4)
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([Block(), Block()])
        self.lm_head = nn.Linear(4, 8)


def test_lora_targets_only_lm_attention():
    from peft import get_peft_model
    pm = get_peft_model(Tiny(), fl.lora_config({}))
    names = fl.trainable_names(pm)
    assert len(names) == 2 * 4 * 2  # layers x projections x (A, B)
    assert all(".self_attn." in n and "language_model" in n for n in names)
    cfg = pm.peft_config["default"]
    assert (cfg.r, cfg.lora_alpha, cfg.lora_dropout) == (16, 32, 0.05)
    pm.base_model.model.model.visual.blocks[0].attn.qkv.weight.requires_grad_(True)
    with pytest.raises(AssertionError):
        fl.trainable_names(pm)
