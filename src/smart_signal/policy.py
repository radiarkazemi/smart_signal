"""Shared trade decision policy for live inference and walk-forward evaluation.

Selectivity over volume: only take BUY/SELL when the model is confident and
directionally decisive. This is the practical lever for win-rate (meta-label
style gating on the primary model's own probabilities).
"""

from __future__ import annotations

from typing import Any

import numpy as np

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


def decide_direction(
    probs: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    expected_log_return: float | None = None,
    ms_bias: float | None = None,
    htf_trend_align: float | None = None,
) -> tuple[int, float, dict[str, float]]:
    """Map class probabilities → SELL/HOLD/BUY with confidence gating.

    Returns ``(class_id, confidence, diagnostics)``.
    """
    inf = (cfg or {}).get("inference") or {}
    hold_thr = float(inf.get("hold_threshold", 0.38))
    min_conf = float(inf.get("min_confidence", 0.48))
    min_edge = float(inf.get("min_edge", 0.15))
    require_ret = bool(inf.get("require_return_align", True))
    require_structure = bool(inf.get("require_structure_align", False))
    structure_min = float(inf.get("structure_min", 0.15))

    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size != 3:
        raise ValueError(f"expected 3 class probs, got shape {p.shape}")
    p = p / max(float(p.sum()), 1e-12)

    p_sell, p_hold, p_buy = float(p[0]), float(p[1]), float(p[2])
    raw = int(np.argmax(p))
    conf = float(p[raw])
    edge = abs(p_buy - p_sell)
    side = 2 if p_buy >= p_sell else 0
    side_conf = float(max(p_buy, p_sell))

    diagnostics = {
        "p_sell": p_sell,
        "p_hold": p_hold,
        "p_buy": p_buy,
        "edge": edge,
        "raw_cls": float(raw),
        "side_conf": side_conf,
    }

    # Soft hold mass or weak directional confidence → stand aside.
    if p_hold >= hold_thr or side_conf < min_conf or edge < min_edge:
        return 1, float(max(p_hold, side_conf)), diagnostics

    # Expected-return head must agree with the chosen side (cheap meta-filter).
    if require_ret and expected_log_return is not None:
        er = float(expected_log_return)
        if side == 2 and er < 0:
            return 1, side_conf, diagnostics
        if side == 0 and er > 0:
            return 1, side_conf, diagnostics

    # Optional ICT / HTF structure agreement (off by default; enable in config).
    if require_structure:
        bias = 0.0 if ms_bias is None else float(ms_bias)
        htf = 0.0 if htf_trend_align is None else float(htf_trend_align)
        struct = 0.6 * bias + 0.4 * htf
        diagnostics["structure"] = struct
        if side == 2 and struct < structure_min:
            return 1, side_conf, diagnostics
        if side == 0 and struct > -structure_min:
            return 1, side_conf, diagnostics

    return side, side_conf, diagnostics


def apply_trade_cooldown(
    rows: list[dict[str, Any]],
    *,
    cooldown_bars: int,
    signal_key: str = "signal",
) -> list[dict[str, Any]]:
    """Force HOLD on overlapping entries so reported win-rate is non-overlapping."""
    if cooldown_bars <= 0:
        return rows
    out: list[dict[str, Any]] = []
    next_free = -10**9
    for i, row in enumerate(rows):
        r = dict(row)
        sig = r.get(signal_key)
        if sig in {"BUY", "SELL"}:
            if i < next_free:
                r[signal_key] = "HOLD"
                if "signal_with_price" in r and "price" in r:
                    r["signal_with_price"] = f"HOLD @ {float(r['price']):.2f}"
                r["win"] = None
                r["cooldown_skipped"] = True
            else:
                next_free = i + int(cooldown_bars)
                r["cooldown_skipped"] = False
        else:
            r["cooldown_skipped"] = False
        out.append(r)
    return out
