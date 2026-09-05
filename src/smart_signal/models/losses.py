from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 1.6,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    n_cls = logits.size(-1)
    log_p = F.log_softmax(logits, dim=-1)
    if label_smoothing and label_smoothing > 0:
        # Smooth hard labels then apply focal modulation on the true class probability.
        smooth = float(label_smoothing)
        with torch.no_grad():
            true = torch.zeros_like(log_p).scatter_(1, target.view(-1, 1), 1.0)
            true = true * (1.0 - smooth) + smooth / n_cls
        p = log_p.exp()
        pt = (p * true).sum(dim=-1).clamp_min(1e-8)
        loss = -((1.0 - pt).clamp_min(0.0) ** gamma) * (true * log_p).sum(dim=-1)
    else:
        p = log_p.exp()
        pt = p.gather(1, target.view(-1, 1)).squeeze(1)
        log_pt = log_p.gather(1, target.view(-1, 1)).squeeze(1)
        loss = -((1.0 - pt).clamp_min(0.0) ** gamma) * log_pt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    class_weight: torch.Tensor | None,
    gamma: float,
    return_w: float,
    vol_w: float,
    candle_w: float = 0.55,
    path_w: float = 0.35,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = focal_ce(
        outputs["dir_logits"],
        batch["y_dir"],
        weight=class_weight,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    ret = F.smooth_l1_loss(outputs["y_ret"], batch["y_ret"])
    vol = F.smooth_l1_loss(outputs["y_vol"], batch["y_vol"])

    # Candle direction: weight away from flat so bull/bear matter more.
    candle_counts = torch.bincount(batch["y_candle"], minlength=3).float().clamp_min(1.0)
    candle_weight = (candle_counts.sum() / (3.0 * candle_counts)).to(batch["y_candle"].device)
    candle = focal_ce(
        outputs["candle_logits"],
        batch["y_candle"],
        weight=candle_weight,
        gamma=max(1.0, gamma - 0.2),
        label_smoothing=max(0.0, label_smoothing * 0.5),
    )

    # ATR-scaled OHLC path (stable + geometry-aware).
    path = (
        F.smooth_l1_loss(outputs["y_up"], batch["y_up"])
        + F.smooth_l1_loss(outputs["y_dn"], batch["y_dn"])
        + F.smooth_l1_loss(outputs["y_close_loc"], batch["y_close_loc"])
    ) / 3.0

    total = ce + return_w * ret + vol_w * vol + candle_w * candle + path_w * path
    parts = {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "ret": float(ret.detach().cpu()),
        "vol": float(vol.detach().cpu()),
        "candle": float(candle.detach().cpu()),
        "path": float(path.detach().cpu()),
    }
    return total, parts
