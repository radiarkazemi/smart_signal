from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smart_signal.config import artifacts_dir, checkpoint_path, load_config
from smart_signal.data.dataset import MTFGoldDataset, collate_batch, valid_indices
from smart_signal.models.goldnet import build_goldnet
from smart_signal.train import prepare_frames


@torch.no_grad()
def run_backtest(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    bt = cfg.get("backtest") or {}
    inf = cfg.get("inference") or {}
    label_cfg = cfg.get("label") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = prepare_frames(cfg)
    indices = valid_indices(frames, cfg)
    if len(indices) < 16:
        raise RuntimeError("Not enough windows to backtest")
    cut = int(len(indices) * 0.8)
    holdout = indices[cut:]
    ds = MTFGoldDataset(frames, cfg, holdout)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
    model = build_goldnet(cfg).to(device)
    ckpt = checkpoint_path(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    start = float(bt.get("start_usd", 10000))
    risk = float(bt.get("risk_frac", 0.004))
    spread = float(bt.get("spread_usd", inf.get("spread_usd", 0.35)))
    hold_thr = float(inf.get("hold_threshold", 0.42))
    min_conf = float(inf.get("min_confidence", 0.36))
    sl_atr = float(label_cfg.get("sl_atr", 1.15))
    equity = start
    peak = start
    max_dd = 0.0
    trades: list[dict[str, Any]] = []
    wins = 0
    for batch in loader:
        batch_d = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_d)
        probs = F.softmax(out["dir_logits"], dim=-1).cpu().numpy()
        y = batch["y_dir"].numpy()
        ret = batch["y_ret"].numpy()
        close = batch["close"].numpy()
        for i in range(len(y)):
            p = probs[i]
            cls = int(np.argmax(p))
            conf = float(p[cls])
            if cls == 1 or p[1] >= hold_thr or conf < min_conf:
                continue
            direction = 1.0 if cls == 2 else -1.0
            px = float(close[i])
            exit_px = px * float(np.exp(ret[i]))
            sl_dist = max(px * 0.0015 * sl_atr, 0.4)
            oz = (equity * risk) / sl_dist
            pnl = oz * (direction * (exit_px - px) - spread)
            equity = max(50.0, equity + pnl)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
            hit = cls == int(y[i])
            wins += int(hit and pnl > 0)
            trades.append({"dir": "BUY" if cls == 2 else "SELL", "conf": conf, "pnl": pnl, "hit": hit})

    n = len(trades)
    summary = {
        "n_windows": int(len(holdout)),
        "n_trades": n,
        "win_rate": (wins / n) if n else 0.0,
        "end_equity": round(equity, 2),
        "return_pct": round(100.0 * (equity / start - 1.0), 3),
        "max_drawdown_pct": round(100.0 * max_dd, 3),
        "avg_trade_pnl": round(float(np.mean([t["pnl"] for t in trades])) if trades else 0.0, 3),
        "start_usd": start,
    }
    (artifacts_dir() / "backtest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
