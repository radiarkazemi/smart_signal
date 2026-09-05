from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from smart_signal.config import data_dir, env, load_config
from smart_signal.data.forexcom import cache_forexcom_dump, fetch_last_price, fetch_live_1m_bars
from smart_signal.data.public import cache_public_gold, fetch_yahoo_chart
from smart_signal.train import ingest_cached_raw, train_model


def cmd_fetch(_args: argparse.Namespace) -> None:
    written = ingest_cached_raw()
    try:
        live = fetch_live_1m_bars(limit=500)
        print(f"live 1m bars: {len(live)}")
    except Exception as exc:
        print(f"live bars skipped: {exc}")
    try:
        last = fetch_last_price("xauusd", "1m")
        print(f"last XAUUSD: {last}")
    except Exception as exc:
        print(f"last price skipped: {exc}")
    frames = {}
    try:
        frames["15m"] = fetch_yahoo_chart("GC=F", "15m", "60d")
        frames["1h"] = fetch_yahoo_chart("GC=F", "1h", "730d")
        frames["1d"] = fetch_yahoo_chart("GC=F", "1d", "10y")
        pub = cache_public_gold(frames)
        written.update({str(k): str(v) for v, k in []})
        written.update({str(p): "yahoo" for p in pub.values()})
        print({tf: len(df) for tf, df in frames.items()})
    except Exception as exc:
        print(f"yahoo fetch skipped: {exc}")
    fx_1m = Path("/tmp/xauusd_1m.jsonl")
    if fx_1m.exists():
        cache_forexcom_dump(
            fx_1m,
            Path("/tmp/xauusd_1h.jsonl"),
            Path("/tmp/xauusd.jsonl"),
        )
    print(json.dumps({"cached": written, "data_dir": str(data_dir())}, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    ingest_cached_raw()
    cfg = load_config(args.config)
    metrics = train_model(cfg, epochs=args.epochs)
    slim = {k: v for k, v in metrics.items() if k != "history"}
    print(json.dumps(slim, indent=2, default=str))


def cmd_signal(_args: argparse.Namespace) -> None:
    from smart_signal.infer import SignalEngine, signal_to_dict

    ingest_cached_raw()
    eng = SignalEngine()
    print(json.dumps(signal_to_dict(eng.live_signal()), indent=2))


def cmd_backtest(_args: argparse.Namespace) -> None:
    from smart_signal.backtest import run_backtest

    ingest_cached_raw()
    print(json.dumps(run_backtest(), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    host = args.host or env("SMART_SIGNAL_HOST", "0.0.0.0") or "0.0.0.0"
    port = int(args.port or env("SMART_SIGNAL_PORT", "8080") or 8080)
    uvicorn.run("smart_signal.api:app", host=host, port=port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smart-signal", description="XAUUSD GoldNet signal engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="Pull FOREXCOM + public gold history into data/")
    tr = sub.add_parser("train", help="Train GoldNet")
    tr.add_argument("--config", default=None)
    tr.add_argument("--epochs", type=int, default=None)
    sub.add_parser("signal", help="Print the current live signal")
    sub.add_parser("backtest", help="Walk-forward holdout backtest")
    sv = sub.add_parser("serve", help="Run the signal API + dashboard")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {"fetch": cmd_fetch, "train": cmd_train, "signal": cmd_signal, "backtest": cmd_backtest, "serve": cmd_serve}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
