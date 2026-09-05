"""Next-candle bullish/bearish labels and OHLC targets (causal)."""

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

    y_candle = np.full(n, FLAT, dtype=np.int64)
    y_next_ret = np.zeros(n, dtype=np.float32)
    y_next_high = np.zeros(n, dtype=np.float32)
    y_next_low = np.zeros(n, dtype=np.float32)
    y_next_close = np.zeros(n, dtype=np.float32)

    for i in range(n - 1):
        px = max(c[i], 1e-12)
        nxt_o, nxt_h, nxt_l, nxt_c = o[i + 1], h[i + 1], l[i + 1], c[i + 1]
        body = nxt_c - nxt_o
        rng = max(nxt_h - nxt_l, 1e-12)
        if abs(body) / rng < flat_body_pct:
            y_candle[i] = FLAT
        elif body > 0:
            y_candle[i] = BULL
        else:
            y_candle[i] = BEAR
        y_next_ret[i] = np.log(max(nxt_c, 1e-12) / px)
        y_next_high[i] = np.log(max(nxt_h, 1e-12) / px)
        y_next_low[i] = np.log(max(nxt_l, 1e-12) / px)
        y_next_close[i] = y_next_ret[i]

    # Last bar has no future candle — keep flat / zero targets (excluded by horizon mask).
    out["y_candle"] = y_candle
    out["y_next_ret"] = y_next_ret
    out["y_next_high"] = y_next_high
    out["y_next_low"] = y_next_low
    out["y_next_close"] = y_next_close
    return out
