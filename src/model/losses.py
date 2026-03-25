from typing import Optional

import torch
import torch.nn.functional as F

EPS = 1e-6


def _blend_type_aware_losses(
    cont_loss: Optional[torch.Tensor],
    cat_loss: Optional[torch.Tensor],
    cont_weight: float,
):
    if cont_loss is None and cat_loss is None:
        return None
    if cont_loss is None:
        return cat_loss
    if cat_loss is None:
        return cont_loss

    cont_weight = float(max(0.0, min(1.0, cont_weight)))
    cat_weight = 1.0 - cont_weight
    return cont_weight * cont_loss + cat_weight * cat_loss


def compute_reconstruction_loss(
    model,
    data: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    nask: torch.Tensor,
    cat_logits=None,
):
    target = data.squeeze(dim=1)
    stats = {}

    cont_mask_loss = None
    cont_observed_loss = None
    cat_mask_loss = None
    cat_observed_loss = None

    if len(model.continuous_indices) > 0:
        con_idx = model.continuous_indices
        con_pred = pred.squeeze(dim=2)[:, con_idx]
        con_target = target[:, con_idx]
        con_mask = mask[:, con_idx]
        con_nask = nask[:, con_idx]
        con_sq_error = (con_pred - con_target) ** 2
        cont_mask_loss = (con_sq_error * con_mask).sum() / con_mask.sum().clamp_min(EPS)
        cont_observed_loss = (con_sq_error * con_nask).sum() / con_nask.sum().clamp_min(
            EPS
        )
        stats["cont_loss"] = cont_mask_loss.detach()
    else:
        stats["cont_loss"] = torch.full((), float("nan"), device=target.device)

    if len(model.categorical_indices) > 0 and cat_logits is not None:
        cat_mask_loss_sum = 0.0
        cat_mask_loss_cnt = 0
        cat_observed_loss_sum = 0.0
        cat_observed_loss_cnt = 0
        for idx in model.categorical_indices:
            idx_mask = mask[:, idx] > EPS
            idx_nask = nask[:, idx] > EPS
            if bool(idx_mask.any()):
                logits = cat_logits[idx][idx_mask]
                target_idx = (
                    target[idx_mask, idx].long().clamp(min=0, max=logits.shape[1] - 1)
                )
                cat_mask_loss_sum = cat_mask_loss_sum + F.cross_entropy(
                    logits, target_idx, reduction="mean"
                )
                cat_mask_loss_cnt += 1
            if bool(idx_nask.any()):
                logits = cat_logits[idx][idx_nask]
                target_idx = (
                    target[idx_nask, idx].long().clamp(min=0, max=logits.shape[1] - 1)
                )
                cat_observed_loss_sum = cat_observed_loss_sum + F.cross_entropy(
                    logits, target_idx, reduction="mean"
                )
                cat_observed_loss_cnt += 1

        if cat_mask_loss_cnt > 0:
            cat_mask_loss = cat_mask_loss_sum / float(cat_mask_loss_cnt)
            stats["cat_loss"] = cat_mask_loss.detach()
        else:
            stats["cat_loss"] = torch.full((), float("nan"), device=target.device)

        if cat_observed_loss_cnt > 0:
            cat_observed_loss = cat_observed_loss_sum / float(cat_observed_loss_cnt)
    else:
        stats["cat_loss"] = torch.full((), float("nan"), device=target.device)

    masked_loss = _blend_type_aware_losses(
        cont_mask_loss,
        cat_mask_loss,
        model.type_aware_cont_weight,
    )
    observed_loss = _blend_type_aware_losses(
        cont_observed_loss,
        cat_observed_loss,
        model.type_aware_cont_weight,
    )
    if masked_loss is None or observed_loss is None:
        raise RuntimeError("Type-aware reconstruction received no valid targets.")

    total_loss = masked_loss + model.observed_loss_weight * observed_loss
    stats["masked_loss"] = masked_loss.detach()
    stats["observed_loss"] = observed_loss.detach()
    return total_loss, stats
