from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smart_signal.config import artifacts_dir, checkpoint_path, load_config
from smart_signal.data.dataset import MTFGoldDataset, collate_batch, fit_scaler, time_split, valid_indices
from smart_signal.models.goldnet import build_goldnet
from smart_signal.train import prepare_frames


@torch.no_grad()
def run_backtest(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    bt = cfg.get("backtest") or {}
    inf = cfg.get("inference") or {}
    label_cfg = cfg.get("label") or {}
    train_cfg = cfg.get("train") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = prepare_frames(cfg)
    indices = valid_indices(frames, cfg)
    if len(indices) < 16:
        raise RuntimeError("Not enough windows to backtest")
    import pandas as pd

    t_ns = pd.to_datetime(frames["15m"]["time"], utc=True).astype("int64").to_numpy()
    _train_idx, val_idx = time_split(
        indices,
        t_ns,
        float(train_cfg.get("val_frac", 0.18)),
        int(train_cfg.get("embargo_bars", 8)),
    )
    scaler_state = None
    ckpt = checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    scaler_state = payload.get("scaler")
    from smart_signal.data.dataset import FeatureScaler

    scaler = FeatureScaler.from_state(scaler_state) if scaler_state else fit_scaler(frames, _train_idx)
    ds = MTFGoldDataset(frames, cfg, val_idx, scaler=scaler)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
    model = build_goldnet(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    start = float(bt.get("start_usd", 10000))
    risk = float(bt.get("risk_frac", 0.004))
    spread = float(bt.get("spread_usd", inf.get("spread_usd", 0.35)))
    hold_thr = float(inf.get("hold_threshold", 0.42))
    min_conf = float(inf.get("min_confidence", 0.36))
    sl_atr = float(label_cfg.get("sl_atr", 1.15))
    horizon = int(label_cfg.get("horizon", 8))

    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch_d = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_d)
        probs = F.softmax(out["dir_logits"], dim=-1).cpu().numpy()
        y = batch["y_dir"].numpy()
        ret = batch["y_ret"].numpy()
        close = batch["close"].numpy()
        times = batch["time_ns"].numpy()
        for i in range(len(y)):
            rows.append(
                {
                    "t": int(times[i]),
                    "p": probs[i],
                    "y": int(y[i]),
                    "ret": float(ret[i]),
                    "px": float(close[i]),
                }
            )
    rows.sort(key=lambda r: r["t"])

    equity = start
    peak = start
    max_dd = 0.0
    trades: list[dict[str, Any]] = []
    wins = 0
    last_t = -10**18
    cooldown = 0
    for row in rows:
        if cooldown > 0:
            cooldown -= 1
            continue
        p = row["p"]
        cls = int(np.argmax(p))
        conf = float(p[cls])
        if cls == 1 or p[1] >= hold_thr or conf < min_conf:
            continue
        direction = 1.0 if cls == 2 else -1.0
        px = row["px"]
        exit_px = px * float(np.exp(row["ret"]))
        sl_dist = max(px * 0.0015 * sl_atr, 0.4)
        oz = (equity * risk) / sl_dist
        pnl = oz * (direction * (exit_px - px) - spread)
        equity = max(50.0, equity + pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
        hit = cls == row["y"]
        wins += int(pnl > 0)
        cooldown = horizon
        trades.append({"dir": "BUY" if cls == 2 else "SELL", "conf": conf, "pnl": pnl, "hit": hit})

    n = len(trades)
    summary = {
        "n_windows": int(len(val_idx)),
        "n_trades": n,
        "win_rate": round((wins / n) if n else 0.0, 4),
        "end_equity": round(equity, 2),
        "return_pct": round(100.0 * (equity / start - 1.0), 3),
        "max_drawdown_pct": round(100.0 * max_dd, 3),
        "avg_trade_pnl": round(float(np.mean([t["pnl"] for t in trades])) if trades else 0.0, 3),
        "start_usd": start,
        "val_acc_checkpoint": payload.get("val", {}).get("acc"),
    }
    (artifacts_dir() / "backtest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
