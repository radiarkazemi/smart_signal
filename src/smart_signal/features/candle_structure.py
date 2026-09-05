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


def refine_path_olhc_from_intrabar(bars: pd.DataFrame, bars_child: pd.DataFrame) -> pd.DataFrame:
    """Replace the wick proxy with true low-first vs high-first path from finer intrabar prints (5m/1m).

    For each parent bar, inspect the first time the parent high and low are
    tagged by 1m candles. If low prints before high → OLHC (+1); else OHLC (-1).
    """
    out = bars.copy().reset_index(drop=True)
    if bars_child is None or bars_child.empty or "path_olhc" not in out.columns:
        return out
    child = bars_child[["time", "high", "low"]].copy().sort_values("time").reset_index(drop=True)
    parent = out[["time", "high", "low"]].copy()
    # Parent bar end ≈ next parent open; use asof ranges via searchsorted.
    pt = pd.to_datetime(parent["time"], utc=True).astype("int64").to_numpy()
    ct = pd.to_datetime(child["time"], utc=True).astype("int64").to_numpy()
    ch = child["high"].to_numpy(dtype=np.float64)
    cl = child["low"].to_numpy(dtype=np.float64)
    ph = parent["high"].to_numpy(dtype=np.float64)
    pl = parent["low"].to_numpy(dtype=np.float64)
    path = out["path_olhc"].to_numpy(dtype=np.float64).copy()
    # Assume fixed parent duration from median spacing.
    if len(pt) > 2:
        dur = int(np.median(np.diff(pt)))
    else:
        dur = 15 * 60 * 1_000_000_000
    for i in range(len(parent)):
        start = pt[i]
        end = pt[i] + dur if i + 1 >= len(pt) else pt[i + 1]
        left = int(np.searchsorted(ct, start, side="left"))
        right = int(np.searchsorted(ct, end, side="left"))
        if right - left < 2:
            continue
        hi_i = None
        lo_i = None
        target_h, target_l = ph[i], pl[i]
        for j in range(left, right):
            if hi_i is None and ch[j] >= target_h - 1e-9:
                hi_i = j
            if lo_i is None and cl[j] <= target_l + 1e-9:
                lo_i = j
            if hi_i is not None and lo_i is not None:
                break
        if hi_i is None or lo_i is None:
            continue
        # +1 = low first (OLHC / discount sweep first), -1 = high first
        path[i] = 1.0 if lo_i < hi_i else -1.0
    out["path_olhc"] = path.astype(np.float32)
    return out


def refine_path_olhc_from_1m(bars: pd.DataFrame, bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for 1m intrabar path refinement."""
    return refine_path_olhc_from_intrabar(bars, bars_1m)
