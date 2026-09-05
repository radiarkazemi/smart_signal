from __future__ import annotations

from typing import Any

from smart_signal.config import load_config
from smart_signal.walkforward import run_walkforward


def run_backtest(
    cfg: dict[str, Any] | None = None,
    *,
    holdout_days: int = 2,
    epochs: int | None = None,
    use_production_checkpoint: bool = True,
) -> dict[str, Any]:
    """Holdout backtest using the production model by default."""
    cfg = cfg or load_config()
    result = run_walkforward(
        cfg,
        holdout_days=holdout_days,
        epochs=epochs,
        use_production_checkpoint=use_production_checkpoint,
    )
    summary = result["summary"]
    print(
        f"Holdout days: {summary['holdout_days']}\n"
        f"Side accuracy: {summary.get('side_accuracy', 0):.1%}\n"
        f"Raw direction accuracy: {summary.get('raw_direction_accuracy', 0):.1%}\n"
        f"Trade win rate: {summary['trade_winrate']:.1%} on {summary['n_trades']} trades\n"
        f"Avg pnl/oz: {summary['avg_trade_pnl_usd_per_oz']:.3f}",
        flush=True,
    )
    return result
