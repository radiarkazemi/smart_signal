from __future__ import annotations

from typing import Any

from smart_signal.config import load_config
from smart_signal.walkforward import run_walkforward


def run_backtest(
    cfg: dict[str, Any] | None = None,
    *,
    holdout_days: int = 2,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Walk-forward backtest: train on history, score the last 1–2 trading days."""
    cfg = cfg or load_config()
    result = run_walkforward(cfg, holdout_days=holdout_days, epochs=epochs)
    summary = result["summary"]
    print(
        f"Holdout days: {summary['holdout_days']}\n"
        f"Direction accuracy: {summary['direction_accuracy']:.1%}\n"
        f"Trade win rate: {summary['trade_winrate']:.1%} on {summary['n_trades']} trades\n"
        f"Avg pnl/oz: {summary['avg_trade_pnl_usd_per_oz']:.3f}",
        flush=True,
    )
    return result
