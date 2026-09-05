from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smart_signal.config import artifacts_dir, checkpoint_path, load_config
from smart_signal.data.dataset import (
    FeatureScaler,
    MTFGoldDataset,
    collate_batch,
    fit_scaler,
    valid_indices,
)
from smart_signal.labels.next_candle import decode_next_ohlc
from smart_signal.structure_prior import calibrate_candle_probs, calibrate_path
from smart_signal.models.goldnet import build_goldnet, count_parameters
from smart_signal.policy import apply_trade_cooldown, decide_direction
from smart_signal.train import class_weights, prepare_frames, run_epoch, set_seed

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
CANDLE_LABELS = {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}


def trading_days(times: pd.Series, min_bars: int = 20) -> pd.DatetimeIndex:
    days = pd.to_datetime(times, utc=True).dt.floor("D")
    counts = days.value_counts()
    good = counts[counts >= min_bars].sort_index()
    return pd.DatetimeIndex(good.index)


def split_holdout_days(
    indices: np.ndarray,
    times: pd.Series,
    holdout_days: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    days = pd.to_datetime(times, utc=True).dt.floor("D")
    trading = trading_days(times)
    if len(trading) < holdout_days + 2:
        raise RuntimeError(
            f"Need more trading days for walk-forward (have {len(trading)}, need > {holdout_days + 1})"
        )
    hold = list(trading[-holdout_days:])
    hold_set = set(hold)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i in indices:
        i = int(i)
        if days.iloc[i] in hold_set:
            test_idx.append(i)
        else:
            train_idx.append(i)
    if len(train_idx) < 64 or len(test_idx) < 8:
        raise RuntimeError(
            f"Walk-forward split too small: train={len(train_idx)} test={len(test_idx)}"
        )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
        [d.strftime("%Y-%m-%d") for d in hold],
    )


def train_before_holdout(
    frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    train_indices: np.ndarray,
    *,
    epochs: int,
    checkpoint: Path,
) -> dict[str, Any]:
    train_cfg = cfg.get("train") or {}
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cut = int(len(train_indices) * 0.85)
    cut = min(max(cut, 32), len(train_indices) - 8)
    fit_idx = train_indices[:cut]
    val_idx = train_indices[cut:]

    scaler = fit_scaler(frames, fit_idx)
    train_ds = MTFGoldDataset(frames, cfg, fit_idx, scaler=scaler)
    val_ds = MTFGoldDataset(frames, cfg, val_idx, scaler=scaler)
    y_train = frames["15m"]["y_dir"].to_numpy()[fit_idx]
    class_n = np.bincount(y_train, minlength=3).astype(np.float64)
    sample_w = 1.0 / np.maximum(class_n[y_train], 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    loader_kw = dict(
        batch_size=int(train_cfg.get("batch_size", 48)),
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    model = build_goldnet(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2.8e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 4e-4)),
    )
    n_epochs = max(1, epochs)
    warmup = max(1, int(n_epochs * float(train_cfg.get("warmup_frac", 0.08))))

    def lr_at(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, n_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    weights = class_weights(y_train, device)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    stale = 0
    patience = int(train_cfg.get("patience", 5))
    history: list[dict[str, Any]] = []
    for epoch in range(1, n_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer=opt, cfg=cfg, device=device, weights=weights)
        va = run_epoch(model, val_loader, optimizer=None, cfg=cfg, device=device, weights=weights)
        sched.step()
        history.append({"epoch": epoch, "train": tr, "val": va})
        score = float(va["acc"] + float((cfg.get("train") or {}).get("candle_ckpt_w", 0.15)) * va.get("candle_acc", 0.0))
        if score > best_acc + 0.002:
            best_acc = score
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
                    "scaler": scaler.state_dict(),
                    "val": va,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                    "n_params": count_parameters(model),
                },
                checkpoint,
            )
        else:
            stale += 1
        print(
            f"[walkforward] epoch {epoch:02d}  train_acc={tr['acc']:.3f} "
            f"candle={tr.get('candle_acc', 0):.3f}  "
            f"val_acc={va['acc']:.3f} candle={va.get('candle_acc', 0):.3f}  "
            f"best={best_acc:.3f}",
            flush=True,
        )
        if stale >= patience:
            print(f"[walkforward] early stop at epoch {epoch}", flush=True)
            break
    return {
        "best_val_acc": best_acc,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_params": count_parameters(model),
        "checkpoint": str(checkpoint),
        "history": history,
    }


@torch.no_grad()
def predict_holdout(
    frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    test_indices: np.ndarray,
    checkpoint: Path,
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    scaler = FeatureScaler.from_state(payload.get("scaler"))
    model = build_goldnet(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    inf = cfg.get("inference") or {}
    label_cfg = cfg.get("label") or {}
    spread = float((cfg.get("backtest") or {}).get("spread_usd", inf.get("spread_usd", 0.35)))
    horizon = int(label_cfg.get("horizon", 10))
    cooldown = int(inf.get("trade_cooldown_bars", 6))
    struct_candle_s = float(inf.get("structure_candle_strength", 0.0))
    struct_path_s = float(inf.get("structure_path_strength", 0.18))
    struct_min_abs = float(inf.get("structure_min_abs_score", 0.45))

    tp_atr = float(label_cfg.get("tp_atr", label_cfg.get("tp_atr", 1.55)))
    sl_atr = float(label_cfg.get("sl_atr", label_cfg.get("sl_atr", 1.55)))

    ds = MTFGoldDataset(frames, cfg, test_indices, scaler=scaler)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
    sig = frames["15m"]
    atr = sig["atr"].to_numpy(dtype=np.float64) if "atr" in sig.columns else np.zeros(len(sig))
    ms_bias = sig["ms_bias"].to_numpy(dtype=np.float64) if "ms_bias" in sig.columns else np.zeros(len(sig))
    htf_align = (
        sig["htf_trend_align"].to_numpy(dtype=np.float64)
        if "htf_trend_align" in sig.columns
        else np.zeros(len(sig))
    )
    premium_discount = (
        sig["premium_discount"].to_numpy(dtype=np.float64)
        if "premium_discount" in sig.columns
        else np.zeros(len(sig))
    )
    times = pd.to_datetime(sig["time"], utc=True)
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    # Map each dataset row back to its 15m bar index for correct day/ATR lookup.
    bar_by_ns = {int(time_ns[i]): int(i) for i in range(len(time_ns))}

    rows: list[dict[str, Any]] = []
    cursor = 0
    for batch in loader:
        batch_d = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_d)
        probs = F.softmax(out["dir_logits"], dim=-1).cpu().numpy()
        candle_probs = F.softmax(out["candle_logits"], dim=-1).cpu().numpy()
        y = batch["y_dir"].numpy()
        y_candle = batch["y_candle"].numpy()
        ret = batch["y_ret"].numpy()
        close = batch["close"].numpy()
        atr_b = batch["atr"].numpy() if "atr" in batch else None
        y_up_t = batch["y_up"].numpy()
        y_dn_t = batch["y_dn"].numpy()
        y_loc_t = batch["y_close_loc"].numpy()
        pred_up = out["y_up"].cpu().numpy()
        pred_dn = out["y_dn"].cpu().numpy()
        pred_loc = out["y_close_loc"].cpu().numpy()
        pred_ret = out["y_ret"].cpu().numpy()
        t_ns = batch["time_ns"].numpy()
        for i in range(len(y)):
            p = probs[i]
            price = float(close[i])
            # Prefer the original bar index from the holdout split order.
            src_i = int(test_indices[cursor + i]) if cursor + i < len(test_indices) else None
            if src_i is None or src_i < 0 or src_i >= len(sig):
                src_i = bar_by_ns.get(int(t_ns[i]))
            if src_i is None:
                src_i = int(np.searchsorted(time_ns, int(t_ns[i]), side="left"))
                src_i = min(max(src_i, 0), len(sig) - 1)
            ts = pd.Timestamp(times.iloc[src_i])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            atr_v = float(atr_b[i]) if atr_b is not None else (
                float(atr[src_i]) if atr[src_i] > 0 else price * 0.002
            )
            raw_cls = int(np.argmax(p))
            side_cls = 2 if float(p[2]) >= float(p[0]) else 0
            cls, conf, diag = decide_direction(
                p,
                cfg=cfg,
                expected_log_return=float(pred_ret[i]),
                ms_bias=float(ms_bias[src_i]),
                htf_trend_align=float(htf_align[src_i]),
            )
            signal = LABELS[cls]
            if signal == "BUY":
                tp = price + tp_atr * atr_v
                sl = price - sl_atr * atr_v
            elif signal == "SELL":
                tp = price - tp_atr * atr_v
                sl = price + sl_atr * atr_v
            else:
                tp = price
                sl = price
            actual = LABELS[int(y[i])]
            direction = 1.0 if cls == 2 else (-1.0 if cls == 0 else 0.0)
            exit_px = price * float(np.exp(ret[i]))
            pnl = direction * (exit_px - price) - (spread if direction != 0 else 0.0)

            # ICT/MTF prior teaches next-candle bull/bear + OHLC path at inference.
            cal_candle = calibrate_candle_probs(
                candle_probs[i],
                ms_bias=float(ms_bias[src_i]),
                htf_trend_align=float(htf_align[src_i]),
                premium_discount=float(premium_discount[src_i]),
                strength=struct_candle_s,
                min_abs_score=struct_min_abs,
            )
            candle_cls = int(np.argmax(cal_candle))
            true_candle = int(y_candle[i])
            cal_up, cal_dn, cal_loc = calibrate_path(
                float(pred_up[i]),
                float(pred_dn[i]),
                float(pred_loc[i]),
                ms_bias=float(ms_bias[src_i]),
                htf_trend_align=float(htf_align[src_i]),
                premium_discount=float(premium_discount[src_i]),
                strength=struct_path_s,
                min_abs_score=struct_min_abs,
            )
            pred_hi, pred_lo, pred_cl = decode_next_ohlc(
                price, atr_v, cal_up, cal_dn, cal_loc
            )
            _th, _tl, true_cl = decode_next_ohlc(
                price, atr_v, float(y_up_t[i]), float(y_dn_t[i]), float(y_loc_t[i])
            )
            close_mae = abs(pred_cl - true_cl)
            range_width = pred_hi - pred_lo
            rows.append(
                {
                    "time": ts.isoformat(),
                    "day": ts.strftime("%Y-%m-%d"),
                    "signal": signal,
                    "signal_with_price": f"{signal} @ {price:.2f}",
                    "price": round(price, 3),
                    "confidence": round(conf, 4),
                    "p_sell": round(float(p[0]), 4),
                    "p_hold": round(float(p[1]), 4),
                    "p_buy": round(float(p[2]), 4),
                    "edge": round(float(diag.get("edge", 0.0)), 4),
                    "take_profit": round(tp, 3),
                    "stop_loss": round(sl, 3),
                    "actual": actual,
                    "raw_pred": LABELS[raw_cls],
                    "side_pred": LABELS[side_cls],
                    "correct": bool(cls == int(y[i])),
                    "raw_correct": bool(raw_cls == int(y[i])),
                    "side_correct": (bool(side_cls == int(y[i])) if int(y[i]) != 1 else None),
                    "forward_return": round(float(ret[i]), 6),
                    "pnl_usd_per_oz": round(float(pnl), 3),
                    "win": bool(pnl > 0) if signal in {"BUY", "SELL"} else None,
                    "horizon_bars": horizon,
                    "candle_pred": CANDLE_LABELS[candle_cls],
                    "candle_actual": CANDLE_LABELS[true_candle],
                    "candle_correct": bool(candle_cls == true_candle),
                    "pred_next_high": round(pred_hi, 3),
                    "pred_next_low": round(pred_lo, 3),
                    "pred_next_close": round(pred_cl, 3),
                    "true_next_close": round(true_cl, 3),
                    "close_mae": round(float(close_mae), 3),
                    "pred_range_width": round(float(range_width), 3),
                    "ohlc_ok": bool(pred_lo <= pred_cl <= pred_hi),
                }
            )
        cursor += len(y)
    rows.sort(key=lambda r: r["time"])
    rows = apply_trade_cooldown(rows, cooldown_bars=cooldown, signal_key="signal")
    # Recompute trade outcomes after cooldown may have forced HOLD.
    for r in rows:
        if r["signal"] not in {"BUY", "SELL"}:
            r["win"] = None
            r["pnl_usd_per_oz"] = 0.0
            r["signal_with_price"] = f"HOLD @ {float(r['price']):.2f}"
        else:
            direction = 1.0 if r["signal"] == "BUY" else -1.0
            exit_px = float(r["price"]) * float(np.exp(r["forward_return"]))
            pnl = direction * (exit_px - float(r["price"])) - spread
            r["pnl_usd_per_oz"] = round(float(pnl), 3)
            r["win"] = bool(pnl > 0)
            r["signal_with_price"] = f"{r['signal']} @ {float(r['price']):.2f}"
    return rows



def summarize(rows: list[dict[str, Any]], holdout_days: list[str]) -> dict[str, Any]:
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    n_raw = sum(1 for r in rows if r.get("raw_correct"))
    side_rows = [r for r in rows if r.get("side_correct") is not None]
    n_side = sum(1 for r in side_rows if r.get("side_correct"))
    trades = [r for r in rows if r["signal"] in {"BUY", "SELL"}]
    wins = [r for r in trades if r.get("win")]
    candle_n = sum(1 for r in rows if r.get("candle_correct") is not None)
    candle_ok = sum(1 for r in rows if r.get("candle_correct"))
    bull_bear_rows = [r for r in rows if r.get("candle_actual") in {"BULLISH", "BEARISH"}]
    bull_bear_ok = sum(1 for r in bull_bear_rows if r.get("candle_correct"))
    close_maes = [float(r["close_mae"]) for r in rows if r.get("close_mae") is not None]
    widths = [float(r["pred_range_width"]) for r in rows if r.get("pred_range_width") is not None]
    by_day: dict[str, Any] = {}
    for day in holdout_days:
        day_rows = [r for r in rows if r["day"] == day]
        day_trades = [r for r in day_rows if r["signal"] in {"BUY", "SELL"}]
        day_wins = [r for r in day_trades if r.get("win")]
        day_side = [r for r in day_rows if r.get("side_correct") is not None]
        day_bb = [r for r in day_rows if r.get("candle_actual") in {"BULLISH", "BEARISH"}]
        by_day[day] = {
            "bars": len(day_rows),
            "direction_accuracy": round(
                (sum(1 for r in day_rows if r["correct"]) / len(day_rows)) if day_rows else 0.0,
                4,
            ),
            "raw_direction_accuracy": round(
                (sum(1 for r in day_rows if r.get("raw_correct")) / len(day_rows)) if day_rows else 0.0,
                4,
            ),
            "side_accuracy": round(
                (sum(1 for r in day_side if r.get("side_correct")) / len(day_side)) if day_side else 0.0,
                4,
            ),
            "candle_accuracy": round(
                (sum(1 for r in day_rows if r.get("candle_correct")) / len(day_rows)) if day_rows else 0.0,
                4,
            ),
            "bull_bear_accuracy": round(
                (sum(1 for r in day_bb if r.get("candle_correct")) / len(day_bb)) if day_bb else 0.0,
                4,
            ),
            "trades": len(day_trades),
            "trade_winrate": round((len(day_wins) / len(day_trades)) if day_trades else 0.0, 4),
            "avg_pnl_usd_per_oz": round(
                float(np.mean([r["pnl_usd_per_oz"] for r in day_trades])) if day_trades else 0.0,
                3,
            ),
            "avg_close_mae": round(
                float(np.mean([r["close_mae"] for r in day_rows])) if day_rows else 0.0,
                3,
            ),
        }
    return {
        "holdout_days": holdout_days,
        "n_bars": n,
        "direction_accuracy": round((n_correct / n) if n else 0.0, 4),
        "raw_direction_accuracy": round((n_raw / n) if n else 0.0, 4),
        "side_accuracy": round((n_side / len(side_rows)) if side_rows else 0.0, 4),
        "candle_accuracy": round((candle_ok / candle_n) if candle_n else 0.0, 4),
        "bull_bear_accuracy": round((bull_bear_ok / len(bull_bear_rows)) if bull_bear_rows else 0.0, 4),
        "avg_close_mae": round(float(np.mean(close_maes)) if close_maes else 0.0, 3),
        "avg_pred_range_width": round(float(np.mean(widths)) if widths else 0.0, 3),
        "ohlc_ok_rate": round((sum(1 for r in rows if r.get("ohlc_ok")) / n) if n else 0.0, 4),
        "n_trades": len(trades),
        "trade_winrate": round((len(wins) / len(trades)) if trades else 0.0, 4),
        "avg_trade_pnl_usd_per_oz": round(
            float(np.mean([r["pnl_usd_per_oz"] for r in trades])) if trades else 0.0,
            3,
        ),
        "buy_signals": sum(1 for r in rows if r["signal"] == "BUY"),
        "sell_signals": sum(1 for r in rows if r["signal"] == "SELL"),
        "hold_signals": sum(1 for r in rows if r["signal"] == "HOLD"),
        "by_day": by_day,
    }


def run_walkforward(
    cfg: dict[str, Any] | None = None,
    *,
    holdout_days: int = 2,
    epochs: int | None = None,
    use_production_checkpoint: bool = True,
) -> dict[str, Any]:
    """Score holdout days with priced signals.

    Default evaluates the production checkpoint (live model). Pass
    ``use_production_checkpoint=False`` for a purged retrain-before-holdout run.
    """
    cfg = cfg or load_config()
    train_cfg = cfg.get("train") or {}
    n_hold = max(1, int(holdout_days))
    n_epochs = int(epochs if epochs is not None else min(18, int(train_cfg.get("epochs", 28))))

    frames = prepare_frames(cfg)
    indices = valid_indices(frames, cfg)
    train_idx, test_idx, hold_days = split_holdout_days(indices, frames["15m"]["time"], n_hold)
    print(
        f"[walkforward] train bars={len(train_idx)}  holdout days={hold_days}  test bars={len(test_idx)}",
        flush=True,
    )

    out_dir = artifacts_dir()
    if use_production_checkpoint:
        ckpt = Path(checkpoint_path(cfg))
        if not ckpt.exists():
            raise FileNotFoundError(f"Production checkpoint missing: {ckpt}")
        print(f"[walkforward] evaluating production checkpoint {ckpt}", flush=True)
        train_info: dict[str, Any] = {
            "best_val_acc": None,
            "n_params": None,
            "checkpoint": str(ckpt),
        }
    else:
        ckpt = out_dir / "walkforward_goldnet.pt"
        train_info = train_before_holdout(frames, cfg, train_idx, epochs=n_epochs, checkpoint=ckpt)

    rows = predict_holdout(frames, cfg, test_idx, ckpt)
    summary = summarize(rows, hold_days)
    summary.update(
        {
            "n_train_bars": int(len(train_idx)),
            "n_test_bars": int(len(test_idx)),
            "train_val_acc": train_info.get("best_val_acc"),
            "n_params": train_info.get("n_params"),
            "checkpoint": train_info["checkpoint"],
            "mode": "production_holdout" if use_production_checkpoint else "purged_retrain",
        }
    )

    sample = rows[:: max(1, len(rows) // 40)][:40]
    out = {
        "summary": summary,
        "signals": sample,
        "all_signals_path": str(out_dir / "walkforward_signals.jsonl"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "walkforward.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    signals_path = out_dir / "walkforward_signals.jsonl"
    with signals_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")

    (out_dir / "backtest.json").write_text(
        json.dumps(
            {
                "mode": summary.get("mode", "walkforward_holdout_days"),
                "n_windows": summary["n_test_bars"],
                "n_trades": summary["n_trades"],
                "win_rate": summary["trade_winrate"],
                "direction_accuracy": summary["direction_accuracy"],
                "raw_direction_accuracy": summary.get("raw_direction_accuracy"),
                "side_accuracy": summary.get("side_accuracy"),
                "candle_accuracy": summary.get("candle_accuracy"),
                "avg_close_mae": summary.get("avg_close_mae"),
                "avg_pred_range_width": summary.get("avg_pred_range_width"),
                "ohlc_ok_rate": summary.get("ohlc_ok_rate"),
                "holdout_days": summary["holdout_days"],
                "avg_trade_pnl": summary["avg_trade_pnl_usd_per_oz"],
                "train_val_acc": summary.get("train_val_acc"),
                "by_day": summary["by_day"],
                "sample_signals": sample[:12],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
