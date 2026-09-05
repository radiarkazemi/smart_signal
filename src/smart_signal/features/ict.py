"""ICT-inspired market-structure features (causal / no lookahead).

Concepts encoded as numeric features the network can learn from:
  - swing highs / lows → market structure bias
  - BOS / CHOCH (break / change of character)
  - Fair Value Gaps (FVG) distance & fill state
  - Order-block proximity
  - liquidity sweeps of equal highs / lows
  - premium / discount of price inside the recent dealing range
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ICT_COLUMNS = [
    "ms_bias",
    "bos",
    "choch",
    "fvg_up",
    "fvg_down",
    "fvg_mid_dist",
    "ob_bull_dist",
    "ob_bear_dist",
    "liq_sweep_hi",
    "liq_sweep_lo",
    "eq_highs",
    "eq_lows",
    "premium_discount",
    "disp_up",
    "disp_down",
]


def _atr_like(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.empty_like(tr)
    out[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out


def add_ict_features(df: pd.DataFrame, swing: int = 3) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    for col in ICT_COLUMNS:
        out[col] = 0.0
    if n < swing * 2 + 3:
        return out

    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    atr = _atr_like(h, l, c, 14)
    eps = np.maximum(atr * 0.15, c * 1e-5)

    swing_hi = np.zeros(n, dtype=bool)
    swing_lo = np.zeros(n, dtype=bool)
    for i in range(swing, n - swing):
        window_h = h[i - swing : i + swing + 1]
        window_l = l[i - swing : i + swing + 1]
        if h[i] >= window_h.max() - 1e-12:
            swing_hi[i] = True
        if l[i] <= window_l.min() + 1e-12:
            swing_lo[i] = True

    # Confirmed swings are only known after `swing` bars — shift forward so
    # feature at time t uses swings confirmed by t (no lookahead).
    conf_hi = np.roll(swing_hi, swing)
    conf_lo = np.roll(swing_lo, swing)
    conf_hi[: swing * 2] = False
    conf_lo[: swing * 2] = False

    last_sh_px = np.full(n, np.nan)
    last_sl_px = np.full(n, np.nan)
    sh = np.nan
    sl = np.nan
    for i in range(n):
        if conf_hi[i]:
            sh = h[i - swing]
        if conf_lo[i]:
            sl = l[i - swing]
        last_sh_px[i] = sh
        last_sl_px[i] = sl

    ms_bias = np.zeros(n, dtype=np.float64)
    bos = np.zeros(n, dtype=np.float64)
    choch = np.zeros(n, dtype=np.float64)
    bias = 0.0
    for i in range(1, n):
        if np.isnan(last_sh_px[i - 1]) or np.isnan(last_sl_px[i - 1]):
            ms_bias[i] = bias
            continue
        broke_hi = c[i] > last_sh_px[i - 1] + eps[i]
        broke_lo = c[i] < last_sl_px[i - 1] - eps[i]
        if broke_hi:
            if bias <= 0:
                choch[i] = 1.0
            else:
                bos[i] = 1.0
            bias = 1.0
        elif broke_lo:
            if bias >= 0:
                choch[i] = -1.0
            else:
                bos[i] = -1.0
            bias = -1.0
        ms_bias[i] = bias

    # Fair value gaps: candle[i-2].high < candle[i].low (bullish) etc.
    fvg_up = np.zeros(n, dtype=np.float64)
    fvg_down = np.zeros(n, dtype=np.float64)
    fvg_mid_dist = np.zeros(n, dtype=np.float64)
    active_up = np.nan
    active_dn = np.nan
    for i in range(2, n):
        if l[i] > h[i - 2] + eps[i]:
            mid = 0.5 * (l[i] + h[i - 2])
            active_up = mid
            fvg_up[i] = (l[i] - h[i - 2]) / np.maximum(c[i], 1e-12)
        if h[i] < l[i - 2] - eps[i]:
            mid = 0.5 * (h[i] + l[i - 2])
            active_dn = mid
            fvg_down[i] = (l[i - 2] - h[i]) / np.maximum(c[i], 1e-12)
        # Fill / invalidate
        if not np.isnan(active_up) and l[i] <= active_up:
            active_up = np.nan
        if not np.isnan(active_dn) and h[i] >= active_dn:
            active_dn = np.nan
        if not np.isnan(active_up):
            fvg_mid_dist[i] = (c[i] - active_up) / np.maximum(c[i], 1e-12)
        elif not np.isnan(active_dn):
            fvg_mid_dist[i] = (c[i] - active_dn) / np.maximum(c[i], 1e-12)

    # Order blocks: last opposing candle before displacement.
    ob_bull = np.nan
    ob_bear = np.nan
    ob_bull_dist = np.zeros(n, dtype=np.float64)
    ob_bear_dist = np.zeros(n, dtype=np.float64)
    disp_up = np.zeros(n, dtype=np.float64)
    disp_down = np.zeros(n, dtype=np.float64)
    body = c - o
    for i in range(2, n):
        move = (c[i] - c[i - 2]) / np.maximum(atr[i], 1e-12)
        if move > 1.6 and body[i] > 0:
            # bullish displacement — mark prior down candle as bullish OB
            for j in range(i - 1, max(-1, i - 6), -1):
                if c[j] < o[j]:
                    ob_bull = 0.5 * (o[j] + c[j])
                    break
            disp_up[i] = np.clip(move / 3.0, 0.0, 1.0)
        if move < -1.6 and body[i] < 0:
            for j in range(i - 1, max(-1, i - 6), -1):
                if c[j] > o[j]:
                    ob_bear = 0.5 * (o[j] + c[j])
                    break
            disp_down[i] = np.clip(-move / 3.0, 0.0, 1.0)
        if not np.isnan(ob_bull):
            ob_bull_dist[i] = (c[i] - ob_bull) / np.maximum(c[i], 1e-12)
            if c[i] < ob_bull - atr[i]:
                ob_bull = np.nan
        if not np.isnan(ob_bear):
            ob_bear_dist[i] = (c[i] - ob_bear) / np.maximum(c[i], 1e-12)
            if c[i] > ob_bear + atr[i]:
                ob_bear = np.nan

    # Equal highs / lows + liquidity sweeps.
    eq_highs = np.zeros(n, dtype=np.float64)
    eq_lows = np.zeros(n, dtype=np.float64)
    liq_sweep_hi = np.zeros(n, dtype=np.float64)
    liq_sweep_lo = np.zeros(n, dtype=np.float64)
    recent_hi: list[float] = []
    recent_lo: list[float] = []
    for i in range(n):
        if conf_hi[i]:
            px = h[i - swing]
            for prev in recent_hi[-3:]:
                if abs(px - prev) <= eps[i]:
                    eq_highs[i] = 1.0
                    break
            recent_hi.append(px)
            if len(recent_hi) > 8:
                recent_hi = recent_hi[-8:]
        if conf_lo[i]:
            px = l[i - swing]
            for prev in recent_lo[-3:]:
                if abs(px - prev) <= eps[i]:
                    eq_lows[i] = 1.0
                    break
            recent_lo.append(px)
            if len(recent_lo) > 8:
                recent_lo = recent_lo[-8:]
        if recent_hi and h[i] > max(recent_hi[-3:]) + eps[i] * 0.25 and c[i] < max(recent_hi[-3:]):
            liq_sweep_hi[i] = 1.0
        if recent_lo and l[i] < min(recent_lo[-3:]) - eps[i] * 0.25 and c[i] > min(recent_lo[-3:]):
            liq_sweep_lo[i] = 1.0

    # Premium / discount inside last confirmed dealing range.
    premium_discount = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(last_sh_px[i]) or np.isnan(last_sl_px[i]):
            continue
        hi, lo = last_sh_px[i], last_sl_px[i]
        if hi <= lo:
            continue
        # 0 = discount extreme, 1 = premium extreme; center at 0.5 → map to [-1, 1]
        loc = (c[i] - lo) / (hi - lo)
        premium_discount[i] = float(np.clip(loc * 2.0 - 1.0, -1.0, 1.0))

    out["ms_bias"] = ms_bias
    out["bos"] = bos
    out["choch"] = choch
    out["fvg_up"] = fvg_up
    out["fvg_down"] = fvg_down
    out["fvg_mid_dist"] = fvg_mid_dist
    out["ob_bull_dist"] = ob_bull_dist
    out["ob_bear_dist"] = ob_bear_dist
    out["liq_sweep_hi"] = liq_sweep_hi
    out["liq_sweep_lo"] = liq_sweep_lo
    out["eq_highs"] = eq_highs
    out["eq_lows"] = eq_lows
    out["premium_discount"] = premium_discount
    out["disp_up"] = disp_up
    out["disp_down"] = disp_down
    return out
