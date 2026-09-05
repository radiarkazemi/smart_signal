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
from smart_signal.labels.next_candle import decode_next_ohlc
from smart_signal.structure_prior import calibrate_candle_probs, calibrate_path
from smart_signal.candle_blend import blend_candle_probs
from smart_signal.models.goldnet import build_goldnet
from smart_signal.policy import decide_direction

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
CANDLE_LABELS = {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}


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
    candle_bias: str = "FLAT"
    p_bear: float = 0.0
    p_flat: float = 0.0
    p_bull: float = 0.0
    pred_next_high: float = 0.0
    pred_next_low: float = 0.0
    pred_next_close: float = 0.0
    ict_summary: str = ""
    ohlc_ok: bool = True
    pred_range_width: float = 0.0


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
        merged = False
        if "1h" in extras:
            frames["1h"] = _merge_tf(frames["1h"], extras["1h"])
            merged = True
        if "4h" in extras:
            frames["4h"] = _merge_tf(frames["4h"], extras["4h"])
            merged = True
        if "1d" in extras:
            frames["1d"] = _merge_tf(frames["1d"], extras["1d"])
            merged = True
        if merged:
            from smart_signal.data.ohlcv import resample_ohlcv
            from smart_signal.features.indicators import add_features
            from smart_signal.features.mtf_align import attach_mtf_alignment

            if "1d" in frames and not frames["1d"].empty:
                frames["1w"] = add_features(
                    resample_ohlcv(
                        frames["1d"][["time", "open", "high", "low", "close", "volume"]], "1w"
                    )
                )
            frames = attach_mtf_alignment(frames)
        # Prefer true OLHC/OHLC path from 1m prints when available (teaches candle structure).
        if "15m" in frames and df_1m is not None and not df_1m.empty:
            from smart_signal.features.candle_structure import refine_path_olhc_from_1m

            frames["15m"] = refine_path_olhc_from_1m(frames["15m"], df_1m)
            if "path_olhc" in frames["15m"].columns:
                frames["15m"]["path_olhc"] = (
                    frames["15m"]["path_olhc"]
                    .replace([float("inf"), float("-inf")], float("nan"))
                    .fillna(0.0)
                    .astype("float32")
                )
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
        exp_ret = float(out["y_ret"].squeeze().cpu())
        exp_vol = float(out["y_vol"].squeeze().cpu())
        price = float(frames["15m"]["close"].iloc[-1])
        atr = float(frames["15m"]["atr"].iloc[-1]) if "atr" in frames["15m"].columns else price * 0.002
        label_cfg = self.cfg.get("label") or {}
        row = frames["15m"].iloc[-1]
        inf = self.cfg.get("inference") or {}
        struct_candle_s = float(inf.get("structure_candle_strength", 0.0))
        struct_path_s = float(inf.get("structure_path_strength", 0.18))
        struct_min_abs = float(inf.get("structure_min_abs_score", 0.45))
        cls, conf, _diag = decide_direction(
            probs,
            cfg=self.cfg,
            expected_log_return=exp_ret,
            ms_bias=float(row.get("ms_bias", 0.0) or 0.0),
            htf_trend_align=float(row.get("htf_trend_align", 0.0) or 0.0),
        )
        signal = LABELS[cls]
        if signal == "BUY":
            tp = price + float(label_cfg.get("tp_atr", 1.55)) * atr
            sl = price - float(label_cfg.get("sl_atr", 1.55)) * atr
        elif signal == "SELL":
            tp = price - float(label_cfg.get("tp_atr", 1.55)) * atr
            sl = price + float(label_cfg.get("sl_atr", 1.55)) * atr
        else:
            tp = price
            sl = price
        reasons = _explain(out, probs, signal)
        candle_probs = F.softmax(out["candle_logits"], dim=-1).squeeze(0).cpu().numpy()
        ms = float(row.get("ms_bias", 0.0) or 0.0)
        htf = float(row.get("htf_trend_align", 0.0) or 0.0)
        pd_loc = float(row.get("premium_discount", 0.0) or 0.0)
        candle_probs = calibrate_candle_probs(
            candle_probs,
            ms_bias=ms,
            htf_trend_align=htf,
            premium_discount=pd_loc,
            strength=struct_candle_s,
            min_abs_score=struct_min_abs,
        )
        candle_probs = blend_candle_probs(candle_probs, row)
        candle_cls = int(np.argmax(candle_probs))
        y_up = float(out["y_up"].squeeze().cpu())
        y_dn = float(out["y_dn"].squeeze().cpu())
        y_close_loc = float(out["y_close_loc"].squeeze().cpu())
        y_up, y_dn, y_close_loc = calibrate_path(
            y_up,
            y_dn,
            y_close_loc,
            ms_bias=ms,
            htf_trend_align=htf,
            premium_discount=pd_loc,
            strength=struct_path_s,
            min_abs_score=struct_min_abs,
        )
        pred_hi, pred_lo, pred_cl = decode_next_ohlc(price, atr, y_up, y_dn, y_close_loc)
        ohlc_ok = bool(pred_lo <= pred_cl <= pred_hi)
        range_width = max(pred_hi - pred_lo, 0.0)
        ict_summary = _ict_snapshot(frames["15m"])
        reasons.append(
            f"Next candle {CANDLE_LABELS[candle_cls]} "
            f"(bear={candle_probs[0]:.0%} flat={candle_probs[1]:.0%} bull={candle_probs[2]:.0%})"
        )
        reasons.append(
            f"Predicted next range {pred_lo:.2f} – {pred_hi:.2f} "
            f"(close≈{pred_cl:.2f}, width={range_width:.2f})"
        )
        if ict_summary:
            reasons.append(ict_summary)
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
            candle_bias=CANDLE_LABELS[candle_cls],
            p_bear=round(float(candle_probs[0]), 4),
            p_flat=round(float(candle_probs[1]), 4),
            p_bull=round(float(candle_probs[2]), 4),
            pred_next_high=round(pred_hi, 3),
            pred_next_low=round(pred_lo, 3),
            pred_next_close=round(pred_cl, 3),
            ict_summary=ict_summary,
            ohlc_ok=ohlc_ok,
            pred_range_width=round(range_width, 3),
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
        "Features include candle OHLC path, ICT structure (FVG/OB/BOS), and nested MTF alignment",
    ]
    if "vsn_15m" in out:
        w = out["vsn_15m"].squeeze(0).cpu().numpy()
        top = np.argsort(w)[-3:][::-1]
        names = [FEATURE_COLUMNS[i] if i < len(FEATURE_COLUMNS) else f"f{i}" for i in top]
        reasons.append("15m variable selection: " + ", ".join(names))
    return reasons


def _ict_snapshot(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    row = frame.iloc[-1]
    bits = []
    bias = float(row.get("ms_bias", 0.0) or 0.0)
    if bias > 0.2:
        bits.append("bullish structure")
    elif bias < -0.2:
        bits.append("bearish structure")
    if abs(float(row.get("bos", 0.0) or 0.0)) > 0.5:
        bits.append("BOS")
    if abs(float(row.get("choch", 0.0) or 0.0)) > 0.5:
        bits.append("CHOCH")
    if float(row.get("fvg_up", 0.0) or 0.0) > 0:
        bits.append("bullish FVG")
    if float(row.get("fvg_down", 0.0) or 0.0) > 0:
        bits.append("bearish FVG")
    if float(row.get("liq_sweep_hi", 0.0) or 0.0) > 0.5:
        bits.append("liquidity sweep high")
    if float(row.get("liq_sweep_lo", 0.0) or 0.0) > 0.5:
        bits.append("liquidity sweep low")
    pd_ = float(row.get("premium_discount", 0.0) or 0.0)
    if pd_ > 0.35:
        bits.append("premium zone")
    elif pd_ < -0.35:
        bits.append("discount zone")
    if not bits:
        return ""
    return "ICT now: " + ", ".join(bits)


def signal_to_dict(sig: Signal) -> dict[str, Any]:
    data = asdict(sig)
    data["signal_with_price"] = f"{sig.signal} @ {sig.price:.2f}"
    data["candle_with_range"] = (
        f"{sig.candle_bias} → next {sig.pred_next_low:.2f}/{sig.pred_next_close:.2f}/{sig.pred_next_high:.2f}"
    )
    return data
