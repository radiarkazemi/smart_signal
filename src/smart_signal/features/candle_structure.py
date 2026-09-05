"""Teach the model how a candle is built (OHLC / path geometry).

These features describe *where* the open/close sit inside the range, wick
imbalance, multi-bar patterns, and a simple proxy for whether price probed
liquidity low-first (OLHC-like) or high-first (OHLC-like) inside the bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CANDLE_COLUMNS = [
    "close_loc",
    "open_loc",
    "gap_pct",
    "bull_bear",
    "consec_dir",
    "engulf",
    "inside_bar",
    "outside_bar",
    "doji",
    "pin_up",
    "pin_down",
    "marubozu",
    "wick_imbalance",
    "path_olhc",  # >0 ≈ low probed before high (discount first)
]


def add_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    n = len(out)
    if n == 0:
        for col in CANDLE_COLUMNS:
            out[col] = []
        return out

    rng = np.maximum(h - l, 1e-12)
    body = c - o
    abs_body = np.abs(body)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    close_loc = (c - l) / rng
    open_loc = (o - l) / rng
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    prev_o = np.roll(o, 1)
    prev_o[0] = o[0]
    prev_h = np.roll(h, 1)
    prev_h[0] = h[0]
    prev_l = np.roll(l, 1)
    prev_l[0] = l[0]
    gap_pct = (o - prev_c) / np.maximum(prev_c, 1e-12)

    bull_bear = np.sign(body)
    direction = np.sign(c - prev_c)
    consec = np.zeros(n, dtype=np.float64)
    run = 0.0
    for i in range(n):
        d = direction[i]
        if d == 0:
            run = 0.0
        elif i > 0 and np.sign(run) == d:
            run = run + d
        else:
            run = d
        consec[i] = float(np.clip(run / 5.0, -1.0, 1.0))

    engulf = np.zeros(n, dtype=np.float64)
    bull_eng = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    bear_eng = (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)
    engulf[bull_eng] = 1.0
    engulf[bear_eng] = -1.0
    engulf[0] = 0.0

    inside = ((h <= prev_h) & (l >= prev_l)).astype(np.float64)
    outside = ((h >= prev_h) & (l <= prev_l)).astype(np.float64)
    inside[0] = 0.0
    outside[0] = 0.0

    doji = (abs_body / rng < 0.12).astype(np.float64)
    pin_up = ((upper / rng > 0.55) & (abs_body / rng < 0.28) & (close_loc < 0.45)).astype(np.float64)
    pin_down = ((lower / rng > 0.55) & (abs_body / rng < 0.28) & (close_loc > 0.55)).astype(np.float64)
    marubozu = ((abs_body / rng > 0.82) & (upper / rng < 0.1) & (lower / rng < 0.1)).astype(np.float64) * bull_bear

    wick_imbalance = (lower - upper) / rng
    # Proxy for intra-bar path: stronger lower wick + close in upper half ≈ OLHC
    # (sell-side liquidity taken first); opposite ≈ OHLC.
    path_olhc = np.clip(wick_imbalance + (close_loc - 0.5), -1.0, 1.0)

    out["close_loc"] = close_loc
    out["open_loc"] = open_loc
    out["gap_pct"] = gap_pct
    out["bull_bear"] = bull_bear
    out["consec_dir"] = consec
    out["engulf"] = engulf
    out["inside_bar"] = inside
    out["outside_bar"] = outside
    out["doji"] = doji
    out["pin_up"] = pin_up
    out["pin_down"] = pin_down
    out["marubozu"] = marubozu
    out["wick_imbalance"] = wick_imbalance
    out["path_olhc"] = path_olhc
    return out
