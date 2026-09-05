from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from smart_signal.config import checkpoint_path, load_config
from smart_signal.data.dataset import FeatureScaler, build_timeframes, last_windows
from smart_signal.data.forexcom import fetch_history, fetch_live_1m_bars, fetch_mongo_1m
from smart_signal.data.ohlcv import load_parquet, resample_ohlcv
from smart_signal.config import data_dir
from smart_signal.features.indicators import FEATURE_COLUMNS
from smart_signal.models.goldnet import build_goldnet

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class Signal:
    symbol: str
    timeframe: str
    signal: str
    confidence: float
    p_sell: float
    p_hold: float
    p_buy: float
    expected_log_return: float
    expected_move_usd: float
    expected_volatility: float
    price: float
    take_profit: float
    stop_loss: float
    as_of: str
    model_params: int
    reasons: list[str]


class SignalEngine:
    def __init__(self, cfg: dict[str, Any] | None = None, ckpt_path: Path | None = None) -> None:
        self.cfg = cfg or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_goldnet(self.cfg).to(self.device)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        path = ckpt_path or checkpoint_path(self.cfg)
        self.loaded = False
        self.ckpt_path = path
        self.scaler = None
        self._htf_extra: dict[str, pd.DataFrame] | None = None
        if path.exists():
            payload = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model"])
            self.scaler = FeatureScaler.from_state(payload.get("scaler"))
            self.loaded = True

    def _frames_from_1m(self, df_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
        frames = build_timeframes(base_1m=df_1m)
        extras = self._load_htf_extra()
        if "1h" in extras:
            frames["1h"] = _merge_tf(frames["1h"], extras["1h"])
        if "4h" in extras:
            frames["4h"] = _merge_tf(frames["4h"], extras["4h"])
        if "1d" in extras:
            frames["1d"] = _merge_tf(frames["1d"], extras["1d"])
        return frames

    def _load_htf_extra(self) -> dict[str, pd.DataFrame]:
        if self._htf_extra is not None:
            return self._htf_extra
        extras: dict[str, pd.DataFrame] = {}
        cached_1h = data_dir() / "public" / "gc_1h.parquet"
        cached_1d = data_dir() / "public" / "gc_1d.parquet"
        if cached_1h.exists():
            extra = load_parquet(cached_1h)
            extras["1h"] = extra
            extras["4h"] = resample_ohlcv(extra, "4h")
        if cached_1d.exists():
            extras["1d"] = load_parquet(cached_1d)
        self._htf_extra = extras
        return extras

    @torch.no_grad()
    def predict_frames(self, frames: dict[str, pd.DataFrame]) -> Signal:
        batch = last_windows(frames, self.cfg, scaler=self.scaler)
        if batch is None:
            raise RuntimeError("Not enough bars to form a multi-timeframe window")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(batch, explain=True)
        probs = F.softmax(out["dir_logits"], dim=-1).squeeze(0).cpu().numpy()
        inf = self.cfg.get("inference") or {}
        hold_thr = float(inf.get("hold_threshold", 0.42))
        min_conf = float(inf.get("min_confidence", 0.36))
        label_cfg = self.cfg.get("label") or {}
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        if probs[1] >= hold_thr or conf < min_conf:
            cls = 1
            conf = float(max(probs[1], conf if cls == 1 else 1.0 - conf))
        price = float(frames["15m"]["close"].iloc[-1])
        atr = float(frames["15m"]["atr"].iloc[-1]) if "atr" in frames["15m"].columns else price * 0.002
        exp_ret = float(out["y_ret"].squeeze().cpu())
        exp_vol = float(out["y_vol"].squeeze().cpu())
        signal = LABELS[cls]
        if signal == "BUY":
            tp = price + float(label_cfg.get("tp_atr", 1.75)) * atr
            sl = price - float(label_cfg.get("sl_atr", 1.15)) * atr
        elif signal == "SELL":
            tp = price - float(label_cfg.get("tp_atr", 1.75)) * atr
            sl = price + float(label_cfg.get("sl_atr", 1.15)) * atr
        else:
            tp = price
            sl = price
        reasons = _explain(out, probs, signal)
        as_of = pd.Timestamp(frames["15m"]["time"].iloc[-1]).tz_convert("UTC").isoformat()
        return Signal(
            symbol=str(self.cfg.get("symbol", "XAUUSD")),
            timeframe=str(self.cfg.get("signal_timeframe", "15m")),
            signal=signal,
            confidence=round(conf, 4),
            p_sell=round(float(probs[0]), 4),
            p_hold=round(float(probs[1]), 4),
            p_buy=round(float(probs[2]), 4),
            expected_log_return=round(exp_ret, 6),
            expected_move_usd=round(price * (np.exp(exp_ret) - 1.0), 3),
            expected_volatility=round(exp_vol, 6),
            price=round(price, 3),
            take_profit=round(tp, 3),
            stop_loss=round(sl, 3),
            as_of=as_of,
            model_params=self.n_params,
            reasons=reasons,
        )

    def live_signal(self) -> Signal:
        df_1m = _load_live_1m()
        frames = self._frames_from_1m(df_1m)
        return self.predict_frames(frames)


def _merge_tf(primary: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    from smart_signal.features.indicators import add_features

    extra = extra.copy()
    needed = {"time", "open", "high", "low", "close"}
    if extra.empty or not needed.issubset(extra.columns):
        return primary
    if "volume" not in extra.columns:
        extra["volume"] = 0.0
    merged = pd.concat(
        [extra[["time", "open", "high", "low", "close", "volume"]], primary[["time", "open", "high", "low", "close", "volume"]]],
        ignore_index=True,
    )
    merged = merged.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return add_features(merged)


def _load_live_1m() -> pd.DataFrame:
    cached = data_dir() / "forexcom" / "xauusd_1m.parquet"
    live = pd.DataFrame()
    try:
        live = fetch_mongo_1m(limit=4000)
    except Exception:
        live = pd.DataFrame()
    if live.empty:
        try:
            live = fetch_live_1m_bars(limit=2000)
        except Exception:
            live = pd.DataFrame()
    if cached.exists():
        hist = load_parquet(cached)
        if live.empty:
            return hist
        merged = pd.concat([hist, live], ignore_index=True)
        merged = merged.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        return merged
    if not live.empty:
        return live
    try:
        hist = fetch_history("xauusd", "1m", max_pages=1, limit=2000)
        if not hist.empty:
            return hist
    except Exception:
        pass
    raise RuntimeError("Unable to load live XAUUSD 1m bars from server or cache")


def _explain(out: dict[str, torch.Tensor], probs: np.ndarray, signal: str) -> list[str]:
    reasons = [
        f"Direction mix  sell={probs[0]:.0%} hold={probs[1]:.0%} buy={probs[2]:.0%}",
        f"Net signal {signal} from hierarchical 15m←1h←4h←1d GoldNet",
    ]
    if "vsn_15m" in out:
        w = out["vsn_15m"].squeeze(0).cpu().numpy()
        top = np.argsort(w)[-3:][::-1]
        names = [FEATURE_COLUMNS[i] if i < len(FEATURE_COLUMNS) else f"f{i}" for i in top]
        reasons.append("15m variable selection: " + ", ".join(names))
    return reasons


def signal_to_dict(sig: Signal) -> dict[str, Any]:
    return asdict(sig)
