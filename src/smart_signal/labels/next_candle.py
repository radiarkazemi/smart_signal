"""Next-candle bullish/bearish labels and ATR-scaled OHLC targets (causal).

Targets at bar i describe candle i+1:
  - y_candle: 0=bear, 1=flat, 2=bull (by next body)
  - y_up:     (next_high - close_i) / atr_i   (>=0)
  - y_dn:     (close_i - next_low) / atr_i    (>=0)
  - y_close_loc: where next close sits in next range, in [0, 1]

Decoding (guarantees low <= close <= high):
  high  = close + y_up * atr
  low   = close - y_dn * atr
  close = low + y_close_loc * (high - low)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 0 = next candle bearish, 1 = flat/doji, 2 = next candle bullish
BEAR, FLAT, BULL = 0, 1, 2


def next_candle_labels(df: pd.DataFrame, *, flat_body_pct: float = 0.08) -> pd.DataFrame:
    """Label each closed bar with the *next* candle's direction and OHLC path.

    Targets are shifted so row i predicts candle i+1 (no leakage into features).
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    if "atr" in out.columns:
        atr = np.maximum(out["atr"].to_numpy(dtype=np.float64), c * 1e-4)
    else:
        atr = np.maximum(c * 0.0015, 1e-6)

    y_candle = np.full(n, FLAT, dtype=np.int64)
    y_up = np.zeros(n, dtype=np.float32)
    y_dn = np.zeros(n, dtype=np.float32)
    y_close_loc = np.full(n, 0.5, dtype=np.float32)
    # Keep legacy aliases for older checkpoints / eval helpers.
    y_next_high = np.zeros(n, dtype=np.float32)
    y_next_low = np.zeros(n, dtype=np.float32)
    y_next_close = np.zeros(n, dtype=np.float32)
    y_next_ret = np.zeros(n, dtype=np.float32)

    for i in range(n - 1):
        px = max(c[i], 1e-12)
        width = max(float(atr[i]), px * 1e-4)
        nxt_o, nxt_h, nxt_l, nxt_c = o[i + 1], h[i + 1], l[i + 1], c[i + 1]
        body = nxt_c - nxt_o
        rng = max(nxt_h - nxt_l, 1e-12)
        if abs(body) / rng < flat_body_pct:
            y_candle[i] = FLAT
        elif body > 0:
            y_candle[i] = BULL
        else:
            y_candle[i] = BEAR

        up = max(nxt_h - px, 0.0) / width
        dn = max(px - nxt_l, 0.0) / width
        # Cap extreme outliers (news spikes) so regression stays stable.
        y_up[i] = np.float32(min(up, 8.0))
        y_dn[i] = np.float32(min(dn, 8.0))
        y_close_loc[i] = np.float32(np.clip((nxt_c - nxt_l) / rng, 0.0, 1.0))

        y_next_ret[i] = np.log(max(nxt_c, 1e-12) / px)
        y_next_high[i] = np.log(max(nxt_h, 1e-12) / px)
        y_next_low[i] = np.log(max(nxt_l, 1e-12) / px)
        y_next_close[i] = y_next_ret[i]

    out["y_candle"] = y_candle
    out["y_up"] = y_up
    out["y_dn"] = y_dn
    out["y_close_loc"] = y_close_loc
    out["y_next_ret"] = y_next_ret
    out["y_next_high"] = y_next_high
    out["y_next_low"] = y_next_low
    out["y_next_close"] = y_next_close
    return out


def decode_next_ohlc(
    price: float,
    atr: float,
    y_up: float,
    y_dn: float,
    y_close_loc: float,
    *,
    max_atr_mult: float = 4.0,
) -> tuple[float, float, float]:
    """Decode ATR-scaled heads into next high/low/close with OHLC consistency.

    ``max_atr_mult`` clamps extreme softplus outputs so live ranges stay
    within a few ATRs of price (typical gold 15m move), not hundreds of dollars.
    """
    width = max(float(atr), float(price) * 1e-4, 1e-6)
    cap = max(float(max_atr_mult), 0.5)
    up = float(np.clip(y_up, 0.0, cap))
    dn = float(np.clip(y_dn, 0.0, cap))
    loc = float(np.clip(y_close_loc, 0.0, 1.0))
    high = float(price) + up * width
    low = float(price) - dn * width
    if low > high:
        low, high = high, low
    close = low + loc * max(high - low, 1e-9)
    return high, low, close
