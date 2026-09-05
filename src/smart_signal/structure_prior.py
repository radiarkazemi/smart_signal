"""ICT / nested-MTF structure priors for next-candle teaching at inference.

GoldNet already sees these as features. This module converts the same concepts
into a light post-hoc prior that:
  - nudges bullish/bearish next-candle logits when structure is confident
  - gently shifts close-location / range asymmetry toward ICT discount/premium
    and HTF alignment

The prior is only applied when structure is strong *and* the model is uncertain,
so a confident network head is not overridden.
"""

from __future__ import annotations

import numpy as np


def structure_score(
    ms_bias: float,
    htf_trend_align: float = 0.0,
    premium_discount: float = 0.0,
) -> float:
    """Bullish > 0, bearish < 0. Discount (low PD) favors upside continuation."""
    return float(0.45 * ms_bias + 0.35 * htf_trend_align - 0.20 * premium_discount)


def _entropy3(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def calibrate_candle_probs(
    probs: np.ndarray,
    *,
    ms_bias: float,
    htf_trend_align: float = 0.0,
    premium_discount: float = 0.0,
    strength: float = 0.0,  # off by default; set >0 to enable
    min_abs_score: float = 0.28,
    entropy_gate: float = 0.85,
) -> np.ndarray:
    """Blend model candle probs with an ICT/MTF soft prior when uncertain."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1).copy()
    if p.size != 3:
        return p
    p = np.clip(p, 1e-8, 1.0)
    p /= p.sum()
    score = structure_score(ms_bias, htf_trend_align, premium_discount)
    if abs(score) < min_abs_score:
        return p
    if _entropy3(p) < entropy_gate:
        # Model already confident — leave it alone.
        return p
    # Soft prior over [bear, flat, bull].
    mag = float(np.clip(abs(score), 0.0, 1.0))
    if score > 0:
        prior = np.array([0.15 * (1 - mag), 0.25, 0.60 + 0.25 * mag], dtype=np.float64)
    else:
        prior = np.array([0.60 + 0.25 * mag, 0.25, 0.15 * (1 - mag)], dtype=np.float64)
    prior /= prior.sum()
    alpha = float(np.clip(strength * mag, 0.0, 0.65))
    out = (1.0 - alpha) * p + alpha * prior
    out = np.clip(out, 1e-8, 1.0)
    return out / out.sum()


def calibrate_path(
    y_up: float,
    y_dn: float,
    y_close_loc: float,
    *,
    ms_bias: float,
    htf_trend_align: float = 0.0,
    premium_discount: float = 0.0,
    strength: float = 0.18,
    min_abs_score: float = 0.28,
) -> tuple[float, float, float]:
    """Nudge ATR path geometry toward structure (bull → higher close loc / up>dn)."""
    score = structure_score(ms_bias, htf_trend_align, premium_discount)
    if abs(score) < min_abs_score:
        return float(y_up), float(y_dn), float(np.clip(y_close_loc, 0.0, 1.0))
    mag = float(np.clip(abs(score), 0.0, 1.0))
    alpha = float(np.clip(strength * mag, 0.0, 0.40))
    loc = float(np.clip(y_close_loc, 0.0, 1.0))
    up = max(float(y_up), 0.0)
    dn = max(float(y_dn), 0.0)
    if score > 0:
        target_loc = 0.62 + 0.15 * mag
        up = (1 - alpha) * up + alpha * (up + 0.12 * mag)
        dn = (1 - alpha) * dn + alpha * max(dn - 0.06 * mag, 0.0)
    else:
        target_loc = 0.38 - 0.15 * mag
        dn = (1 - alpha) * dn + alpha * (dn + 0.12 * mag)
        up = (1 - alpha) * up + alpha * max(up - 0.06 * mag, 0.0)
    loc = (1 - alpha) * loc + alpha * target_loc
    return float(up), float(dn), float(np.clip(loc, 0.0, 1.0))
