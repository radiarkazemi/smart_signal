from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from smart_signal.config import ROOT, artifacts_dir, checkpoint_path, env, load_config
from smart_signal.data.forexcom import fetch_quote
from smart_signal.journal import append_signal, latest_signal, signal_history

WEB_DIR = ROOT / "web"
STARTED_AT = datetime.now(timezone.utc).isoformat()
_engine = None
_signal_cache: dict = {"ts": 0.0, "payload": None}
CACHE_SEC = float(env("SMART_SIGNAL_SIGNAL_CACHE_SEC", "45") or 45)


def get_engine():
    global _engine
    if _engine is None:
        from smart_signal.infer import SignalEngine

        _engine = SignalEngine()
    return _engine


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Signal — XAUUSD GoldNet",
        description="Live multi-timeframe gold signals and results dashboard.",
        version="0.1.0",
    )
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def dashboard():
        index = WEB_DIR / "index.html"
        if not index.exists():
            return JSONResponse({"service": "smart-signal", "docs": "/docs"})
        return FileResponse(index)

    @app.get("/health")
    async def health():
        cfg = load_config()
        ckpt = checkpoint_path(cfg)
        loaded = False
        params = None
        device = None
        if _engine is not None:
            loaded = bool(_engine.loaded)
            params = _engine.n_params
            device = str(_engine.device)
        elif ckpt.exists():
            loaded = True
        return {
            "status": "ok",
            "service": "smart-signal",
            "model_loaded": loaded,
            "checkpoint": str(ckpt),
            "checkpoint_exists": ckpt.exists(),
            "params": params,
            "device": device,
            "started_at": STARTED_AT,
        }

    @app.get("/quote")
    async def quote():
        try:
            data = await run_in_threadpool(fetch_quote)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        data["server_time"] = datetime.now(timezone.utc).isoformat()
        return data

    def _compute_signal():
        now = time.time()
        cached = _signal_cache.get("payload")
        if cached and now - float(_signal_cache.get("ts") or 0) < CACHE_SEC:
            return cached
        try:
            eng = get_engine()
            if not eng.loaded:
                raise RuntimeError("Model checkpoint is not loaded. Train first: python -m smart_signal train")
            from smart_signal.infer import signal_to_dict

            payload = signal_to_dict(eng.live_signal())
        except Exception as exc:
            stale = latest_signal()
            if stale:
                stale = dict(stale)
                stale["stale"] = True
                stale["error"] = str(exc)
                return stale
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        stored = append_signal(payload)
        _signal_cache["ts"] = now
        _signal_cache["payload"] = stored
        return stored

    @app.get("/signal")
    async def signal():
        return await run_in_threadpool(_compute_signal)

    @app.get("/signals")
    async def signals(limit: int = 40):
        return {"count": min(limit, 200), "results": signal_history(limit=min(max(limit, 1), 200))}

    @app.get("/metrics")
    async def metrics():
        import json

        out: dict = {}
        path = artifacts_dir() / "backtest.json"
        train = artifacts_dir() / "train_metrics.json"
        if path.exists():
            out["backtest"] = json.loads(path.read_text(encoding="utf-8"))
        if train.exists():
            payload = json.loads(train.read_text(encoding="utf-8"))
            payload.pop("history", None)
            out["train"] = payload
        out["latest_signal"] = latest_signal()
        return out or {"detail": "no metrics yet"}

    @app.get("/status")
    async def status():
        quote_data = None
        quote_error = None
        try:
            quote_data = await run_in_threadpool(fetch_quote)
        except Exception as exc:
            quote_error = str(exc)
        health_body = await health()
        return {
            **health_body,
            "quote": quote_data,
            "quote_error": quote_error,
            "latest_signal": latest_signal(),
            "history_count": len(signal_history(200)),
        }

    return app


app = create_app()
