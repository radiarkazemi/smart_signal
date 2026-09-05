"""Nested multi-timeframe alignment helpers.

Correct nesting for gold (and FX) candles:

  1m ⊂ 5m ⊂ 15m ⊂ 1h ⊂ 4h ⊂ 1d ⊂ 1w

Every higher candle is an exact aggregate of lower candles when clocks are
aligned to exchange boundaries. We teach the model that relationship by
attaching causal as-of context from parent timeframes onto each child bar:
distance to HTF open / mid / high / low, and whether LTF structure agrees
with HTF bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smart_signal.data.ohlcv import resample_ohlcv

ALIGN_PARENTS = {
    "5m": ("15m", "1h", "4h", "1d"),
    "15m": ("1h", "4h", "1d", "1w"),
    "1h": ("4h", "1d", "1w"),
    "4h": ("1d", "1w"),
    "1d": ("1w",),
}

MTF_ALIGN_COLUMNS = [
    "htf_open_dist",
    "htf_mid_dist",
    "htf_range_loc",
    "htf_trend_align",
    "htf_same_dir",
    "parent_body",
    "nested_pos",  # 0..1 progress through parent candle (time alignment)
]


def ensure_alignment_frames(
    frames: dict[str, pd.DataFrame],
    *,
    base_15m: pd.DataFrame | None = None,
    base_1m: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Make sure 5m / 1w exist alongside the core stack."""
    out = dict(frames)
    src_15 = out.get("15m")
    if src_15 is None and base_15m is not None:
        src_15 = base_15m
    if "5m" not in out:
        if base_1m is not None and not base_1m.empty:
            out["5m"] = resample_ohlcv(base_1m, "5m")
        elif src_15 is not None and not src_15.empty:
            # Cannot truly invent 5m from 15m; skip.
            pass
    if "1w" not in out:
        src = out.get("1d")
        if src is None and src_15 is not None:
            src = resample_ohlcv(src_15[["time", "open", "high", "low", "close", "volume"]], "1d")
        if src is not None and not src.empty:
            out["1w"] = resample_ohlcv(src[["time", "open", "high", "low", "close", "volume"]], "1w")
    return out


def _asof_parent(child: pd.DataFrame, parent: pd.DataFrame, prefix: str) -> pd.DataFrame:
    left = child[["time"]].copy()
    right = parent[["time", "open", "high", "low", "close"]].copy()
    right = right.rename(
        columns={
            "open": f"{prefix}_open",
            "high": f"{prefix}_high",
            "low": f"{prefix}_low",
            "close": f"{prefix}_close",
        }
    )
    # merge_asof requires sorted keys; side="backward" = last parent with time <= child
    merged = pd.merge_asof(
        left.sort_values("time"),
        right.sort_values("time"),
        on="time",
        direction="backward",
    )
    return merged


def attach_mtf_alignment(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add nested HTF alignment columns onto each timeframe that has parents."""
    frames = ensure_alignment_frames(frames)
    out: dict[str, pd.DataFrame] = {}
    for tf, df in frames.items():
        if df is None or df.empty:
            out[tf] = df
            continue
        cur = df.copy().reset_index(drop=True)
        parents = ALIGN_PARENTS.get(tf, ())
        # Aggregate across parents into stable column names (nearest / primary parent first).
        agg = {c: np.zeros(len(cur), dtype=np.float64) for c in MTF_ALIGN_COLUMNS}
        weights = 0.0
        for idx, ptf in enumerate(parents):
            parent = frames.get(ptf)
            if parent is None or parent.empty:
                continue
            linked = _asof_parent(cur, parent, "p")
            po = linked["p_open"].to_numpy(dtype=np.float64)
            ph = linked["p_high"].to_numpy(dtype=np.float64)
            pl = linked["p_low"].to_numpy(dtype=np.float64)
            pc = linked["p_close"].to_numpy(dtype=np.float64)
            c = cur["close"].to_numpy(dtype=np.float64)
            o = cur["open"].to_numpy(dtype=np.float64)
            scale = np.maximum(c, 1e-12)
            prng = np.maximum(ph - pl, 1e-12)
            w = 1.0 / (1.0 + idx)  # nearer parent weighs more
            open_dist = (c - po) / scale
            mid = 0.5 * (ph + pl)
            mid_dist = (c - mid) / scale
            range_loc = (c - pl) / prng * 2.0 - 1.0
            parent_body = np.sign(pc - po)
            child_body = np.sign(c - o)
            same = (parent_body * child_body).clip(-1, 1)
            # trend align: child vs parent mid
            trend = np.sign(c - mid) * parent_body
            # nested time position inside parent bar
            pt = pd.to_datetime(linked["time"], utc=True)
            # Approximate parent duration from median parent spacing
            ptimes = pd.to_datetime(parent["time"], utc=True)
            if len(ptimes) > 2:
                dur = float(np.median(np.diff(ptimes.astype("int64").to_numpy())))
            else:
                dur = 1.0
            # find parent start times already in linked via asof (= parent time)
            # Use distance from parent open time encoded in merge — we only have child time.
            # nested_pos ≈ fraction of parent duration elapsed since parent open timestamp.
            # Reconstruct parent open timestamps via another asof on parent time only.
            parent_times = pd.merge_asof(
                cur[["time"]].sort_values("time"),
                parent[["time"]].sort_values("time").rename(columns={"time": "ptime"}),
                left_on="time",
                right_on="ptime",
                direction="backward",
            )["ptime"]
            elapsed = (pt.astype("int64").to_numpy() - pd.to_datetime(parent_times, utc=True).astype("int64").to_numpy()).astype(
                np.float64
            )
            nested = np.clip(elapsed / max(dur, 1.0), 0.0, 1.0)

            agg["htf_open_dist"] += w * np.nan_to_num(open_dist, nan=0.0)
            agg["htf_mid_dist"] += w * np.nan_to_num(mid_dist, nan=0.0)
            agg["htf_range_loc"] += w * np.nan_to_num(range_loc, nan=0.0)
            agg["htf_trend_align"] += w * np.nan_to_num(trend, nan=0.0)
            agg["htf_same_dir"] += w * np.nan_to_num(same, nan=0.0)
            agg["parent_body"] += w * np.nan_to_num(parent_body, nan=0.0)
            agg["nested_pos"] += w * np.nan_to_num(nested, nan=0.0)
            weights += w
        if weights > 0:
            for col in MTF_ALIGN_COLUMNS:
                cur[col] = (agg[col] / weights).astype(np.float32)
        else:
            for col in MTF_ALIGN_COLUMNS:
                cur[col] = np.float32(0.0)
        out[tf] = cur
    return out
