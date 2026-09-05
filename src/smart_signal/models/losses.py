from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 1.6,
) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
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
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = focal_ce(outputs["dir_logits"], batch["y_dir"], weight=class_weight, gamma=gamma)
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
