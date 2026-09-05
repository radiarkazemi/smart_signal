from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from smart_signal.config import artifacts_dir, checkpoint_path, data_dir, load_config, models_dir
from smart_signal.data.dataset import (
    MTFGoldDataset,
    build_timeframes,
    collate_batch,
    label_signal_frame,
    time_split,
    valid_indices,
)
from smart_signal.data.ohlcv import bars_from_yahoo_chart, jsonl_to_frame, load_parquet, save_parquet
from smart_signal.models.goldnet import build_goldnet, count_parameters
from smart_signal.models.losses import multi_task_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_training_15m() -> pd.DataFrame:
    public_15 = data_dir() / "public" / "gc_15m.parquet"
    public_1h = data_dir() / "public" / "gc_1h.parquet"
    fx_1m = data_dir() / "forexcom" / "xauusd_1m.parquet"
    frames = []
    if public_15.exists():
        frames.append(load_parquet(public_15))
    if fx_1m.exists():
        from smart_signal.data.ohlcv import resample_ohlcv

        frames.append(resample_ohlcv(load_parquet(fx_1m), "15m"))
    if not frames and public_1h.exists():
        from smart_signal.data.ohlcv import resample_ohlcv

        # last resort: treat 1h as coarse 15m source by resampling after upsample-free copy
        frames.append(load_parquet(public_1h).rename(columns={}))
    if not frames:
        raise FileNotFoundError(
            "No training data. Run: python -m smart_signal fetch"
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


def _load_training_1m() -> pd.DataFrame | None:
    fx_1m = data_dir() / "forexcom" / "xauusd_1m.parquet"
    if fx_1m.exists():
        return load_parquet(fx_1m)
    return None


def prepare_frames(cfg: dict[str, Any], source_15m: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if source_15m is None:
        source_15m = _load_training_15m()
    frames = build_timeframes(base_15m=source_15m)
    pub = data_dir() / "public"
    from smart_signal.features.indicators import add_features
    from smart_signal.data.ohlcv import resample_ohlcv as _rs

    def _merge(primary: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        cols = ["time", "open", "high", "low", "close", "volume"]
        extra = extra.copy()
        if "volume" not in extra.columns:
            extra["volume"] = 0.0
        merged = pd.concat([extra[cols], primary[cols]], ignore_index=True)
        merged = merged.sort_values("time").drop_duplicates("time", keep="last")
        return add_features(merged.reset_index(drop=True))

    if (pub / "gc_1h.parquet").exists():
        extra_1h = load_parquet(pub / "gc_1h.parquet")
        frames["1h"] = _merge(frames["1h"], extra_1h)
        frames["4h"] = _merge(frames["4h"], _rs(extra_1h, "4h"))
    if (pub / "gc_1d.parquet").exists():
        frames["1d"] = _merge(frames["1d"], load_parquet(pub / "gc_1d.parquet"))
    return label_signal_frame(frames, cfg)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv * (3.0 / inv.sum())
    return torch.tensor(w, dtype=torch.float32, device=device)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict[str, Any],
    device: torch.device,
    weights: torch.Tensor,
) -> dict[str, float]:
    train_cfg = cfg.get("train") or {}
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "ret": 0.0, "vol": 0.0, "acc": 0.0, "n": 0.0}
    correct = 0
    seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, parts = multi_task_loss(
            outputs,
            batch,
            class_weight=weights,
            gamma=float(train_cfg.get("focal_gamma", 1.6)),
            return_w=float(train_cfg.get("return_loss_w", 0.32)),
            vol_w=float(train_cfg.get("vol_loss_w", 0.12)),
        )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("grad_clip", 1.0))
            )
            optimizer.step()
        pred = outputs["dir_logits"].argmax(dim=-1)
        correct += int((pred == batch["y_dir"]).sum().item())
        seen += int(batch["y_dir"].size(0))
        for k in ("loss", "ce", "ret", "vol"):
            totals[k] += parts[k] * batch["y_dir"].size(0)
        totals["n"] += batch["y_dir"].size(0)
    n = max(totals["n"], 1.0)
    return {
        "loss": totals["loss"] / n,
        "ce": totals["ce"] / n,
        "ret": totals["ret"] / n,
        "vol": totals["vol"] / n,
        "acc": correct / max(seen, 1),
    }


def train_model(
    cfg: dict[str, Any] | None = None,
    *,
    source_15m: pd.DataFrame | None = None,
    epochs: int | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    train_cfg = cfg.get("train") or {}
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = prepare_frames(cfg, source_15m=source_15m)
    indices = valid_indices(frames, cfg)
    if len(indices) < 32:
        raise RuntimeError(f"Not enough labeled windows: {len(indices)}")
    times = frames["15m"]["time"]
    # valid_indices returns positions into 15m; split by those timestamps
    t_ns = pd.to_datetime(times, utc=True).astype("int64").to_numpy()
    train_idx, val_idx = time_split(
        indices,
        t_ns,
        float(train_cfg.get("val_frac", 0.18)),
        int(train_cfg.get("embargo_bars", 8)),
    )
    train_ds = MTFGoldDataset(frames, cfg, train_idx)
    val_ds = MTFGoldDataset(frames, cfg, val_idx)
    loader_kw = dict(
        batch_size=int(train_cfg.get("batch_size", 48)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    model = build_goldnet(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2.8e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 4e-4)),
    )
    n_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 18))
    warmup = max(1, int(n_epochs * float(train_cfg.get("warmup_frac", 0.08))))

    def lr_at(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, n_epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    y_train = frames["15m"]["y_dir"].to_numpy()[train_idx]
    weights = class_weights(y_train, device)
    ckpt = checkpoint or checkpoint_path(cfg)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    patience = int(train_cfg.get("patience", 5))
    for epoch in range(1, n_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer=opt, cfg=cfg, device=device, weights=weights)
        va = run_epoch(model, val_loader, optimizer=None, cfg=cfg, device=device, weights=weights)
        sched.step()
        row = {"epoch": epoch, "train": tr, "val": va, "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        improved = va["loss"] < best - 1e-4
        if improved:
            best = va["loss"]
            stale = 0
            payload = {
                "model": model.state_dict(),
                "config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
                "val": va,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_params": count_parameters(model),
                "feature_columns": cfg.get("features"),
            }
            torch.save(payload, ckpt)
        else:
            stale += 1
        print(
            f"epoch {epoch:02d}  train_loss={tr['loss']:.4f} acc={tr['acc']:.3f}  "
            f"val_loss={va['loss']:.4f} acc={va['acc']:.3f}  best={best:.4f}",
            flush=True,
        )
        if stale >= patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break
    metrics = {
        "best_val_loss": best,
        "history": history,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_params": count_parameters(model),
        "checkpoint": str(ckpt),
        "label_counts": {
            "all": frames["15m"]["y_dir"].value_counts().sort_index().to_dict(),
        },
    }
    (artifacts_dir() / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    return metrics


def ingest_cached_raw() -> dict[str, str]:
    """Materialize parquet caches from /tmp dumps or data/raw json."""
    written: dict[str, str] = {}
    fx_dir = data_dir() / "forexcom"
    pub_dir = data_dir() / "public"
    fx_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        Path("/tmp/xauusd_1m.jsonl"): fx_dir / "xauusd_1m.parquet",
        Path("/tmp/xauusd_1h.jsonl"): fx_dir / "xauusd_1h.parquet",
        Path("/tmp/xauusd.jsonl"): fx_dir / "xauusd_1d.parquet",
    }
    for src, dst in mapping.items():
        if src.exists():
            save_parquet(jsonl_to_frame(src), dst)
            written[str(dst)] = str(src)

    yahoo_files = {
        "15m": Path("/tmp/yahoo_15m.json"),
        "1h": Path("/tmp/yahoo_1h_730.json"),
        "1d": Path("/tmp/yahoo_1d.json"),
        "5m": Path("/tmp/yahoo_5m.json"),
    }
    for tf, src in yahoo_files.items():
        if not src.exists():
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        df = bars_from_yahoo_chart(payload)
        dst = pub_dir / f"gc_{tf}.parquet"
        save_parquet(df, dst)
        written[str(dst)] = str(src)
    return written
