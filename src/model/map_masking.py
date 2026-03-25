from typing import Tuple

import torch

EPS = 1e-6


def compute_progress(model) -> float:
    t = min(int(model.global_step.item()), model.total_steps)
    return float(t) / float(model.total_steps)


def compute_masking_state(model) -> Tuple[float, float]:
    progress = compute_progress(model)
    alpha = model.alpha_min + (model.alpha_max - model.alpha_min) * (
        progress**model.gamma
    )
    alpha = float(max(0.0, min(1.0, alpha)))
    rho_t = float(model.rho_max) * alpha
    rho_t = float(max(0.0, min(1.0, rho_t)))
    return alpha, rho_t


def compute_feature_weights(model, m: torch.Tensor):
    observed_rate = m.float().mean(dim=0)
    batch_missing_rate = 1.0 - observed_rate
    global_missing_rate = getattr(model, "global_missing_rate", None)
    if global_missing_rate is None or global_missing_rate.numel() != m.shape[1]:
        missing_rate = batch_missing_rate
    else:
        missing_rate = global_missing_rate.to(device=m.device, dtype=torch.float32)

    safe_missing_rate = missing_rate.clamp(min=EPS, max=1.0 - EPS)
    transformed_rate = torch.log(safe_missing_rate / (1.0 - safe_missing_rate))

    logits = model.weight_a * transformed_rate + model.weight_b
    base_probs = torch.sigmoid(logits)
    return base_probs.clamp_(0.0, 1.0)


def apply_map_masking(model, x: torch.Tensor, m: torch.Tensor):
    n, l, d = x.shape

    if not model.training:
        model.last_alpha_t = 0.0
        model.last_rho_t = 0.0
        observed = m > EPS
        natural_missing = ~observed
        token_bank = model.encoder_mask_token.expand(n, l, d)
        x_masked = torch.where(natural_missing.unsqueeze(-1), token_bank, x)

        mask = natural_missing.float()
        nask = observed.float()
        return x_masked, mask, nask, natural_missing

    alpha_t, rho_t = compute_masking_state(model)
    model.last_alpha_t = alpha_t
    model.last_rho_t = rho_t

    base_probs = compute_feature_weights(model, m)
    remaskable_cols = model.remaskable_feature_mask
    remask_probs = (rho_t * base_probs).clamp_(0.0, 1.0)

    observed = m > EPS
    candidate = observed & remaskable_cols.unsqueeze(0)
    remasked = (
        torch.rand((n, l), device=x.device) < remask_probs.unsqueeze(0)
    ) & candidate

    observed_count = observed.sum(dim=1)
    remasked_count = remasked.sum(dim=1)
    overflow_rows = (observed_count > 0) & (remasked_count >= observed_count)
    if bool(overflow_rows.any()):
        for row_idx in torch.nonzero(overflow_rows, as_tuple=False).squeeze(1).tolist():
            rem_idx = torch.nonzero(remasked[row_idx], as_tuple=False).squeeze(1)
            keep_idx = rem_idx[torch.randint(rem_idx.numel(), (1,), device=x.device)]
            remasked[row_idx, keep_idx] = False

    encoder_mask_pos = remasked | (~observed)
    token_bank = model.encoder_mask_token.expand(n, l, d)
    x_masked = torch.where(encoder_mask_pos.unsqueeze(-1), token_bank, x)
    mask = remasked.float()
    nask = (1.0 - mask) * m

    return x_masked, mask, nask, encoder_mask_pos
