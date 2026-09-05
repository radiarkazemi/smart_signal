from __future__ import annotations

import numpy as np
import pandas as pd

# 0 = sell, 1 = hold, 2 = buy
SELL, HOLD, BUY = 0, 1, 2


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 8,
    tp_atr: float = 1.75,
    sl_atr: float = 1.15,
    atr_col: str = "atr",
    min_atr_pct: float = 0.0004,
) -> pd.DataFrame:
    """Lopez de Prado triple-barrier labels on OHLCV + ATR.

    Upper barrier first → buy, lower barrier first → sell, timeout → hold.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    labels = np.full(n, HOLD, dtype=np.int64)
    fwd_ret = np.zeros(n, dtype=np.float64)
    hit_bar = np.full(n, -1, dtype=np.int64)
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    atr = out[atr_col].to_numpy(dtype=np.float64) if atr_col in out.columns else np.zeros(n)
    last_valid = n - horizon - 1
    for i in range(max(0, last_valid + 1)):
        px = close[i]
        if px <= 0:
            continue
        width = max(float(atr[i]), px * min_atr_pct)
        up = px + tp_atr * width
        down = px - sl_atr * width
        end = min(n, i + 1 + horizon)
        chosen = HOLD
        chosen_j = -1
        for j in range(i + 1, end):
            if high[j] >= up:
                chosen = BUY
                chosen_j = j
                break
            if low[j] <= down:
                chosen = SELL
                chosen_j = j
                break
        labels[i] = chosen
        if chosen_j > i:
            fwd_ret[i] = np.log(close[chosen_j] / px)
            hit_bar[i] = chosen_j - i
        elif end - 1 > i:
            fwd_ret[i] = np.log(close[end - 1] / px)
            hit_bar[i] = end - 1 - i
    out["y_dir"] = labels
    out["y_ret"] = fwd_ret.astype(np.float32)
    out["y_vol"] = np.abs(fwd_ret).astype(np.float32)
    out["y_horizon"] = hit_bar
    out.loc[last_valid + 1 :, ["y_dir"]] = HOLD
    return out
