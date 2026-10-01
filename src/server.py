"""Live inference service (SPEC 10). Same model.py code path and config as predict.py.

uvicorn src.server:app --port 8000          (config from $GEOHAI_CONFIG, default config.yaml)
python src/server.py --port 8000

GET  /health  -> {"status": "ok", "model_id", "revision"}
POST /predict -> one raw_outputs.jsonl-style record, for either
                 multipart upload (field "image")  or  JSON {"trial_id": "S017"}
"""
import argparse
import io
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, countries, load_config  # noqa: E402
from model import image_key, load_temperature, predict_record  # noqa: E402
from predict import load_image  # noqa: E402

SPLITS = ("study", "dev")


def default_model(cfg):
    from model import GeoModel
    return GeoModel(cfg)


def resolve_trial(trial_id, data_dir=DATA):
    """trial_id -> image_path from data/{study,dev}/{split}_meta.csv, or None.
    ponytail: re-reads the 2 small CSVs per request so a rebuilt dataset needs no restart."""
    for split in SPLITS:
        path = Path(data_dir) / split / f"{split}_meta.csv"
        if path.exists():
            meta = pd.read_csv(path, dtype=str, keep_default_na=False)
            hit = meta.loc[meta["trial_id"] == trial_id, "image_path"]
            if len(hit):
                return hit.iloc[0]
    return None


def decode_upload(data):
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception as e:  # PIL raises many types (UnidentifiedImageError, OSError, DecompressionBombError...)
        raise HTTPException(400, f"could not decode image: {e}")


def create_app(cfg=None, model_factory=default_model, data_dir=DATA):
    """model_factory(cfg) -> GeoModel-like object; injectable so tests can skip loading weights."""
    cfg = cfg or load_config(os.environ.get("GEOHAI_CONFIG", ROOT / "config.yaml"))
    srv = cfg.get("server", {})
    max_bytes = int(srv.get("max_upload_mb", 10) * 1024 * 1024)
    labels = countries(cfg)
    lock = threading.Lock()  # ponytail: one model, one request at a time; add a queue/batching if throughput matters

    @asynccontextmanager
    async def lifespan(app):
        app.state.gm = model_factory(cfg)
        app.state.T, app.state.T_source = load_temperature(cfg)
        yield
        app.state.gm = None

    app = FastAPI(title="GeoGuess-HAI", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=srv.get("cors_origins", ["*"]),
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def run(image, trial_id, key):
        with lock:
            rec = predict_record(app.state.gm, image, trial_id, cfg, labels, app.state.T, key=key)
        rec["temperature_source"] = app.state.T_source
        return rec

    @app.get("/health")
    def health():
        gm = app.state.gm
        return {"status": "ok", "model_id": gm.model_id, "revision": gm.revision}

    @app.post("/predict")
    async def predict(request: Request):
        ctype = request.headers.get("content-type", "")
        if int(request.headers.get("content-length") or 0) > max_bytes + 64 * 1024:
            raise HTTPException(413, f"request larger than {max_bytes} bytes")

        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("image")
            if upload is None or isinstance(upload, str):
                raise HTTPException(400, 'multipart body needs a file field "image"')
            data = await upload.read()
            if len(data) > max_bytes:
                raise HTTPException(413, f"image larger than {max_bytes} bytes")
            if not data:
                raise HTTPException(400, "empty image")
            image = decode_upload(data)
            return await run_in_threadpool(run, image, "upload", image_key(data))

        if ctype.startswith("application/json"):
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(400, "invalid JSON body")
            trial_id = body.get("trial_id") if isinstance(body, dict) else None
            if not isinstance(trial_id, str) or not trial_id:
                raise HTTPException(400, 'JSON body must be {"trial_id": "S017"}')
            rel = resolve_trial(trial_id, data_dir)
            if rel is None:
                raise HTTPException(404, f"unknown trial_id {trial_id!r}")
            if not (ROOT / rel).exists():
                raise HTTPException(404, f"image for {trial_id} not found: {rel}")
            image = load_image(rel)
            return await run_in_threadpool(run, image, trial_id, None)

        raise HTTPException(415, 'send multipart/form-data with "image" or application/json with "trial_id"')

    return app


app = create_app()


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", help="config path (default: $GEOHAI_CONFIG or config.yaml)")
    args = ap.parse_args()
    uvicorn.run(create_app(load_config(args.config)) if args.config else app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
