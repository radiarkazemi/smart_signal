from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from smart_signal.data.ohlcv import resample_ohlcv
from smart_signal.features.indicators import FEATURE_COLUMNS, add_features, feature_matrix
from smart_signal.labels.triple_barrier import triple_barrier_labels

SIGNAL_TF = "15m"
HTF_ORDER = ("15m", "1h", "4h", "1d")


def build_timeframes(base_15m: pd.DataFrame | None = None, base_1m: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if base_15m is None or base_15m.empty:
        if base_1m is None or base_1m.empty:
            raise ValueError("Need 15m or 1m OHLCV to build timeframes")
        base_15m = resample_ohlcv(base_1m, "15m")
    frames = {"15m": add_features(base_15m)}
    src = base_15m
    frames["1h"] = add_features(resample_ohlcv(src, "1h"))
    frames["4h"] = add_features(resample_ohlcv(src, "4h"))
    frames["1d"] = add_features(resample_ohlcv(src, "1d"))
    return frames


def label_signal_frame(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    label_cfg = cfg.get("label") or {}
    frames = dict(frames)
    frames["15m"] = triple_barrier_labels(
        frames["15m"],
        horizon=int(label_cfg.get("horizon", 8)),
        tp_atr=float(label_cfg.get("tp_atr", 1.75)),
        sl_atr=float(label_cfg.get("sl_atr", 1.15)),
        min_atr_pct=float(label_cfg.get("min_atr_pct", 0.0004)),
    )
    return frames


def _lookbacks(cfg: dict[str, Any]) -> dict[str, int]:
    tfs = cfg.get("timeframes") or {}
    return {tf: int((tfs.get(tf) or {}).get("lookback", 64)) for tf in HTF_ORDER}


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "FeatureScaler | None":
        if not state:
            return None
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


def fit_scaler(frames: dict[str, pd.DataFrame], indices: np.ndarray) -> FeatureScaler:
    mat = feature_matrix(frames["15m"])[np.asarray(indices, dtype=np.int64)]
    mean = mat.mean(axis=0)
    std = np.maximum(mat.std(axis=0), 1e-6)
    return FeatureScaler(mean.astype(np.float32), std.astype(np.float32))


def _time_ns(frame: pd.DataFrame) -> np.ndarray:
    # Force ns so int64 values are always epoch-nanoseconds (not us/ms).
    s = pd.to_datetime(frame["time"], utc=True)
    return np.asarray(s.to_numpy(dtype="datetime64[ns]").astype(np.int64), dtype=np.int64)


def valid_indices(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> np.ndarray:
    lbs = _lookbacks(cfg)
    horizon = int((cfg.get("label") or {}).get("horizon", 8))
    sig = frames["15m"]
    times = {tf: _time_ns(frames[tf]) for tf in HTF_ORDER}
    n = len(sig)
    good = []
    for i in range(lbs["15m"] - 1, n - horizon - 1):
        t = times["15m"][i]
        ok = True
        for tf in HTF_ORDER[1:]:
            pos = int(np.searchsorted(times[tf], t, side="right") - 1)
            if pos < lbs[tf] - 1:
                ok = False
                break
        if ok:
            good.append(i)
    return np.asarray(good, dtype=np.int64)


def time_split(indices: np.ndarray, times: np.ndarray, val_frac: float, embargo: int) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 10:
        cut = max(1, len(indices) - max(1, len(indices) // 5))
        return indices[:cut], indices[cut:]
    order = np.argsort(times[indices])
    ordered = indices[order]
    cut = int(len(ordered) * (1.0 - val_frac))
    cut = min(max(cut, 1), len(ordered) - 1)
    train = ordered[: max(1, cut - embargo)]
    val = ordered[min(len(ordered), cut + embargo) :]
    if len(val) == 0:
        val = ordered[-max(1, len(ordered) // 8) :]
        train = ordered[: -len(val)]
    return train, val


@dataclass
class SampleTensors:
    x: dict[str, torch.Tensor]
    y_dir: torch.Tensor
    y_ret: torch.Tensor
    y_vol: torch.Tensor
    close: torch.Tensor
    time_ns: torch.Tensor


class MTFGoldDataset(Dataset):
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        cfg: dict[str, Any],
        indices: np.ndarray,
        scaler: FeatureScaler | None = None,
    ) -> None:
        self.cfg = cfg
        self.lookbacks = _lookbacks(cfg)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.times = {tf: _time_ns(frames[tf]) for tf in HTF_ORDER}
        self.scaler = scaler
        self.feat = {tf: self._scale(feature_matrix(frames[tf])) for tf in HTF_ORDER}
        sig = frames["15m"]
        self.y_dir = sig["y_dir"].to_numpy(dtype=np.int64)
        self.y_ret = sig["y_ret"].to_numpy(dtype=np.float32)
        self.y_vol = sig["y_vol"].to_numpy(dtype=np.float32)
        self.close = sig["close"].to_numpy(dtype=np.float32)
        self.sig_times = _time_ns(sig)

    def _scale(self, mat: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return mat
        return self.scaler.transform(mat)

    def __len__(self) -> int:
        return int(len(self.indices))

    def _window(self, tf: str, pos: int) -> np.ndarray:
        lb = self.lookbacks[tf]
        sl = self.feat[tf][pos - lb + 1 : pos + 1]
        if sl.shape[0] != lb:
            pad = np.zeros((lb, sl.shape[1] if sl.size else len(FEATURE_COLUMNS)), dtype=np.float32)
            pad[-sl.shape[0] :] = sl
            sl = pad
        return np.ascontiguousarray(sl)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        t = int(self.sig_times[i])
        xs: dict[str, torch.Tensor] = {}
        xs["15m"] = torch.from_numpy(self._window("15m", i))
        for tf in HTF_ORDER[1:]:
            pos = int(np.searchsorted(self.times[tf], t, side="right") - 1)
            pos = max(self.lookbacks[tf] - 1, pos)
            xs[tf] = torch.from_numpy(self._window(tf, pos))
        return {
            "x_15m": xs["15m"],
            "x_1h": xs["1h"],
            "x_4h": xs["4h"],
            "x_1d": xs["1d"],
            "y_dir": torch.tensor(self.y_dir[i], dtype=torch.long),
            "y_ret": torch.tensor(self.y_ret[i], dtype=torch.float32),
            "y_vol": torch.tensor(self.y_vol[i], dtype=torch.float32),
            "close": torch.tensor(self.close[i], dtype=torch.float32),
            "time_ns": torch.tensor(t, dtype=torch.int64),
        }


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {k: torch.stack([it[k] for it in items], dim=0) for k in keys}


def last_windows(
    frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    scaler: FeatureScaler | None = None,
) -> dict[str, torch.Tensor] | None:
    lbs = _lookbacks(cfg)
    xs: dict[str, torch.Tensor] = {}
    t = int(_time_ns(frames["15m"])[-1])
    for tf in HTF_ORDER:
        feat = feature_matrix(frames[tf])
        if scaler is not None:
            feat = scaler.transform(feat)
        times = _time_ns(frames[tf])
        pos = int(np.searchsorted(times, t, side="right") - 1)
        if pos < lbs[tf] - 1:
            return None
        sl = np.ascontiguousarray(feat[pos - lbs[tf] + 1 : pos + 1])
        xs[f"x_{tf}"] = torch.from_numpy(sl).unsqueeze(0)
    return xs
