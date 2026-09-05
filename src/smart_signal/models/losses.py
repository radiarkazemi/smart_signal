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


def _candle_path_consistency(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Tie bullish/bearish candle class to OHLC geometry (close location + range asymmetry)."""
    y = batch["y_candle"]
    loc = outputs["y_close_loc"]
    up = outputs["y_up"]
    dn = outputs["y_dn"]
    target_loc = torch.where(
        y == 2,
        torch.full_like(loc, 0.72),
        torch.where(y == 0, torch.full_like(loc, 0.28), torch.full_like(loc, 0.50)),
    )
    loc_loss = F.smooth_l1_loss(loc, target_loc)
    asymmetric = up - dn
    signed = torch.where(
        y == 2, asymmetric, torch.where(y == 0, -asymmetric, torch.zeros_like(asymmetric))
    )
    mask = (y != 1).float()
    if float(mask.sum()) < 1.0:
        asym_loss = loc.new_zeros(())
    else:
        asym_loss = (F.relu(0.15 - signed) * mask).sum() / mask.sum().clamp_min(1.0)
    probs = F.softmax(outputs["candle_logits"], dim=-1)
    loc_score = (loc - 0.5) * 2.0
    pred_score = probs[:, 2] - probs[:, 0]
    agree = F.smooth_l1_loss(pred_score, loc_score.detach())
    return loc_loss + 0.5 * asym_loss + 0.35 * agree


def _decoded_close_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize next-close error in ATR units after OHLC decode (realtime price teaching)."""
    if "atr" not in batch or "close" not in batch:
        return outputs["y_up"].new_zeros(())
    atr = batch["atr"].clamp_min(1e-6)
    px = batch["close"]
    pred_hi = px + outputs["y_up"] * atr
    pred_lo = px - outputs["y_dn"] * atr
    hi = torch.maximum(pred_hi, pred_lo)
    lo = torch.minimum(pred_hi, pred_lo)
    pred_cl = lo + outputs["y_close_loc"] * (hi - lo).clamp_min(1e-9)
    true_hi = px + batch["y_up"] * atr
    true_lo = px - batch["y_dn"] * atr
    thi = torch.maximum(true_hi, true_lo)
    tlo = torch.minimum(true_hi, true_lo)
    true_cl = tlo + batch["y_close_loc"] * (thi - tlo).clamp_min(1e-9)
    return F.smooth_l1_loss((pred_cl - true_cl) / atr, torch.zeros_like(pred_cl))


def _ict_structure_prior(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """When ICT/MTF structure is confident, gently align next-candle logits with it."""
    if "y_struct" not in batch:
        return outputs["candle_logits"].new_zeros(())
    y = batch["y_struct"].long()
    mask = y != 1
    if int(mask.sum().item()) < 1:
        return outputs["candle_logits"].new_zeros(())
    return focal_ce(
        outputs["candle_logits"][mask],
        y[mask],
        weight=None,
        gamma=1.2,
        label_smoothing=0.05,
    )


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
    consistency_w: float = 0.25,
    price_w: float = 0.35,
    struct_w: float = 0.12,
    label_smoothing: float = 0.0,
    dir_w: float = 1.0,
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

    candle_counts = torch.bincount(batch["y_candle"], minlength=3).float().clamp_min(1.0)
    candle_weight = (candle_counts.sum() / (3.0 * candle_counts)).to(batch["y_candle"].device)
    candle_weight = candle_weight.clone()
    candle_weight[0] = candle_weight[0] * 1.15
    candle_weight[2] = candle_weight[2] * 1.15
    candle = focal_ce(
        outputs["candle_logits"],
        batch["y_candle"],
        weight=candle_weight,
        gamma=max(1.0, gamma - 0.2),
        label_smoothing=max(0.0, label_smoothing * 0.5),
    )

    path = (
        F.smooth_l1_loss(outputs["y_up"], batch["y_up"])
        + F.smooth_l1_loss(outputs["y_dn"], batch["y_dn"])
        + F.smooth_l1_loss(outputs["y_close_loc"], batch["y_close_loc"])
    ) / 3.0

    consistency = _candle_path_consistency(outputs, batch)
    price = _decoded_close_loss(outputs, batch)
    struct = _ict_structure_prior(outputs, batch)

    total = (
        dir_w * ce
        + return_w * ret
        + vol_w * vol
        + candle_w * candle
        + path_w * path
        + consistency_w * consistency
        + price_w * price
        + struct_w * struct
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "ret": float(ret.detach().cpu()),
        "vol": float(vol.detach().cpu()),
        "candle": float(candle.detach().cpu()),
        "path": float(path.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "price": float(price.detach().cpu()),
        "struct": float(struct.detach().cpu()),
    }
    return total, parts
