from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from smart_signal.config import ROOT, artifacts_dir, checkpoint_path, load_config
from smart_signal.infer import SignalEngine, signal_to_dict

WEB_DIR = ROOT / "web"
_engine: SignalEngine | None = None


def get_engine() -> SignalEngine:
    global _engine
    if _engine is None:
        _engine = SignalEngine()
    return _engine


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Signal — XAUUSD GoldNet",
        description="Multi-timeframe deep learning buy/sell signals for gold.",
        version="0.1.0",
    )
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def dashboard():
        index = WEB_DIR / "index.html"
        if not index.exists():
            return JSONResponse({"service": "smart-signal", "docs": "/docs"})
        return FileResponse(index)

    @app.get("/health")
    def health():
        cfg = load_config()
        ckpt = checkpoint_path(cfg)
        engine = get_engine()
        return {
            "status": "ok",
            "model_loaded": engine.loaded,
            "checkpoint": str(ckpt),
            "params": engine.n_params,
            "device": str(engine.device),
        }

    @app.get("/signal")
    def signal():
        try:
            eng = get_engine()
            if not eng.loaded:
                raise RuntimeError("Model checkpoint is not loaded. Train first: python -m smart_signal train")
            sig = eng.live_signal()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return signal_to_dict(sig)

    @app.get("/metrics")
    def metrics():
        path = artifacts_dir() / "backtest.json"
        train = artifacts_dir() / "train_metrics.json"
        out = {}
        if path.exists():
            import json

            out["backtest"] = json.loads(path.read_text(encoding="utf-8"))
        if train.exists():
            import json

            payload = json.loads(train.read_text(encoding="utf-8"))
            payload.pop("history", None)
            out["train"] = payload
        return out or {"detail": "no metrics yet"}

    return app


app = create_app()
