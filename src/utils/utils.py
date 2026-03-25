import math

import numpy as np
import torch
from torch import nn


class MixedFeatureEmbed(nn.Module):
    def __init__(
        self,
        rec_len: int,
        embed_dim: int,
        categorical_feature_mask: torch.Tensor,
        categorical_cardinalities,
        norm_layer=None,
    ):
        super().__init__()
        if categorical_feature_mask.dim() != 1 or int(
            categorical_feature_mask.numel()
        ) != int(rec_len):
            raise ValueError("categorical_feature_mask shape mismatch with rec_len")
        if len(categorical_cardinalities) != rec_len:
            raise ValueError("categorical_cardinalities length mismatch with rec_len")

        self.rec_len = rec_len
        self.embed_dim = embed_dim

        self.register_buffer(
            "categorical_feature_mask",
            categorical_feature_mask.bool(),
            persistent=False,
        )
        self.con_proj = nn.Linear(1, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.cat_embeddings = nn.ModuleDict()
        for idx, is_cat in enumerate(self.categorical_feature_mask.tolist()):
            if is_cat:
                card = max(int(categorical_cardinalities[idx]), 1)
                self.cat_embeddings[str(idx)] = nn.Embedding(card, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 1 or x.shape[2] != self.rec_len:
            raise ValueError(
                f"Expected x shape [B,1,{self.rec_len}], got {tuple(x.shape)}"
            )

        values = x.squeeze(1)  # [B, L]
        tokens = self.con_proj(values.unsqueeze(-1))  # [B, L, D]

        cat_indices = (
            torch.nonzero(self.categorical_feature_mask, as_tuple=False)
            .squeeze(1)
            .tolist()
        )
        for idx in cat_indices:
            emb = self.cat_embeddings[str(idx)]
            ids = values[:, idx].long().clamp(min=0, max=emb.num_embeddings - 1)
            tokens[:, idx, :] = emb(ids)

        return self.norm(tokens)


def get_1d_sincos_pos_embed(
    embed_dim: int, pos: int, cls_token: bool = False
) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin-cos positional embeddings")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = np.arange(pos)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    lr: float,
    min_lr: float,
    max_epochs: int,
    warmup_epochs: int,
) -> float:
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0
            + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs))
        )

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr

    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    norm_type = float(norm_type)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters)

    norms = [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
    return torch.norm(torch.stack(norms), norm_type)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, device_type: str = "cuda", enabled: bool = True):
        self._scaler = torch.amp.GradScaler(device_type, enabled=enabled)

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad=None,
        parameters=None,
        create_graph: bool = False,
        update_grad: bool = True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if not update_grad:
            return None

        if clip_grad is not None:
            if parameters is None:
                raise ValueError("parameters must be provided when clip_grad is set")
            self._scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            self._scaler.unscale_(optimizer)
            norm = get_grad_norm_(parameters)

        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
