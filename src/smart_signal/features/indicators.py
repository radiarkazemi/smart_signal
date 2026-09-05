from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_ret",
    "rng_pct",
    "body_pct",
    "upper_wick",
    "lower_wick",
    "rsi_14",
    "rsi_28",
    "macd",
    "macd_sig",
    "macd_hist",
    "bb_pct",
    "bb_width",
    "atr_pct",
    "stoch_k",
    "stoch_d",
    "ema_gap_fast",
    "ema_gap_slow",
    "rv_20",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rma(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    alpha = 1.0 / period
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, 0.0, None)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    a = high - low
    b = np.abs(high - prev_close)
    c = np.abs(low - prev_close)
    return np.maximum(np.maximum(a, b), c)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    return _rma(true_range(high, low, close), period)


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def bollinger(close: np.ndarray, period: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(close)
    ma = s.rolling(period, min_periods=1).mean().to_numpy()
    sd = s.rolling(period, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    width = (2.0 * k * sd) / np.maximum(ma, 1e-12)
    pct = (close - (ma - k * sd)) / np.maximum(2.0 * k * sd, 1e-12)
    return pct.astype(np.float64), width.astype(np.float64)


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple[np.ndarray, np.ndarray]:
    hh = pd.Series(high).rolling(period, min_periods=1).max().to_numpy()
    ll = pd.Series(low).rolling(period, min_periods=1).min().to_numpy()
    k = 100.0 * (close - ll) / np.maximum(hh - ll, 1e-12)
    d = pd.Series(k).rolling(3, min_periods=1).mean().to_numpy()
    return k, d


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        for col in FEATURE_COLUMNS + ["atr"]:
            out[col] = []
        return out
    out = df.copy().reset_index(drop=True)
    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    prev = np.roll(c, 1)
    prev[0] = c[0]
    log_ret = np.log(np.maximum(c, 1e-12) / np.maximum(prev, 1e-12))
    rng = np.maximum(h - l, 1e-12)
    body = c - o
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    macd_line, macd_sig, macd_hist = macd(c)
    bb_pct, bb_width = bollinger(c)
    atr_v = atr(h, l, c, 14)
    stoch_k, stoch_d = stochastic(h, l, c)
    ema_f = _ema(c, 12)
    ema_s = _ema(c, 48)
    rv = pd.Series(log_ret).rolling(20, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    ts = pd.to_datetime(out["time"], utc=True)
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    dow = ts.dt.dayofweek.to_numpy().astype(np.float64)
    out["log_ret"] = log_ret
    out["rng_pct"] = rng / np.maximum(c, 1e-12)
    out["body_pct"] = body / rng
    out["upper_wick"] = upper / rng
    out["lower_wick"] = lower / rng
    out["rsi_14"] = (rsi(c, 14) - 50.0) / 50.0
    out["rsi_28"] = (rsi(c, 28) - 50.0) / 50.0
    scale = np.maximum(c, 1e-12)
    out["macd"] = macd_line / scale
    out["macd_sig"] = macd_sig / scale
    out["macd_hist"] = macd_hist / scale
    out["bb_pct"] = bb_pct * 2.0 - 1.0
    out["bb_width"] = bb_width
    out["atr"] = atr_v
    out["atr_pct"] = atr_v / scale
    out["stoch_k"] = (stoch_k - 50.0) / 50.0
    out["stoch_d"] = (stoch_d - 50.0) / 50.0
    out["ema_gap_fast"] = (c - ema_f) / scale
    out["ema_gap_slow"] = (c - ema_s) / scale
    out["rv_20"] = rv
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    feat = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[FEATURE_COLUMNS] = feat.astype(np.float32)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
