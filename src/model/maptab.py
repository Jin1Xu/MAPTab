import torch
import torch.nn as nn

from .map_masking import (
    apply_map_masking,
)
from .losses import compute_reconstruction_loss
from .modules import (
    build_model_components,
    configure_feature_types,
    initialize_model_weights,
    run_decoder,
    run_encoder,
)


class MAPMaskingAutoencoder(nn.Module):

    def __init__(
        self,
        rec_len: int = 25,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        alpha_min: float = 0.1,
        alpha_max: float = 1.0,
        gamma: float = 0.5,
        rho_max: float = 0.7,
        weight_a: float = 0.5,
        weight_b: float = 0.2,
        observed_loss_weight: float = 1.0,
        type_aware_cont_weight: float = 0.8,
        total_steps: int = 1000,
        categorical_feature_mask=None,
        categorical_cardinalities=None,
    ):

        super().__init__()
        self.rec_len = int(rec_len)

        configure_feature_types(
            self,
            rec_len=self.rec_len,
            categorical_feature_mask=categorical_feature_mask,
            categorical_cardinalities=categorical_cardinalities,
        )
        build_model_components(
            self,
            rec_len=self.rec_len,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.gamma = float(gamma)
        self.rho_max = float(rho_max)
        self.weight_a = float(weight_a)
        self.weight_b = float(weight_b)
        self.observed_loss_weight = float(observed_loss_weight)
        self.type_aware_cont_weight = float(type_aware_cont_weight)
        self.total_steps = max(int(total_steps), 1)

        self.register_buffer(
            "global_step", torch.zeros(1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "remaskable_feature_mask",
            torch.ones(self.rec_len, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "global_missing_rate",
            torch.zeros(self.rec_len, dtype=torch.float32),
            persistent=True,
        )
        self.last_alpha_t = 0.0
        self.last_rho_t = 0.0

        initialize_model_weights(self)

    def initialize_weights(self):
        initialize_model_weights(self)

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = max(int(total_steps), 1)

    def reset_progress(self) -> None:
        self.global_step.zero_()

    def set_remaskable_features(self, feature_mask: torch.Tensor) -> None:
        if feature_mask.dim() != 1:
            raise ValueError(
                f"feature_mask must be 1-D, got shape={tuple(feature_mask.shape)}"
            )
        if feature_mask.numel() != self.rec_len:
            raise ValueError(
                f"feature_mask length mismatch: expected {self.rec_len}, got {feature_mask.numel()}"
            )
        feature_mask = feature_mask.to(
            device=self.remaskable_feature_mask.device, dtype=torch.bool
        )
        self.remaskable_feature_mask.copy_(feature_mask)

    def set_global_missing_rate(self, missing_rate: torch.Tensor) -> None:
        if missing_rate.dim() != 1:
            raise ValueError(
                f"missing_rate must be 1-D, got shape={tuple(missing_rate.shape)}"
            )
        if missing_rate.numel() != self.rec_len:
            raise ValueError(
                f"missing_rate length mismatch: expected {self.rec_len}, got {missing_rate.numel()}"
            )
        missing_rate = missing_rate.to(
            device=self.global_missing_rate.device, dtype=torch.float32
        )
        self.global_missing_rate.copy_(missing_rate.clamp(0.0, 1.0))

    def map_masking(self, x: torch.Tensor, m: torch.Tensor):
        return apply_map_masking(self, x, m)

    def forward_encoder(self, x: torch.Tensor, m: torch.Tensor):
        return run_encoder(self, x, m)

    def forward_decoder(self, x: torch.Tensor):
        return run_decoder(self, x)

    def reconstruct(self, data: torch.Tensor, miss_idx: torch.Tensor):
        latent, _, _ = self.forward_encoder(data, miss_idx)
        return self.forward_decoder(latent)

    def forward_loss(
        self,
        data: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        nask: torch.Tensor,
        cat_logits=None,
    ):
        return compute_reconstruction_loss(self, data, pred, mask, nask, cat_logits)

    def forward(
        self,
        data: torch.Tensor,
        miss_idx: torch.Tensor,
        return_stats: bool = False,
    ):
        latent, mask, nask = self.forward_encoder(data, miss_idx)
        pred, cat_logits = self.forward_decoder(latent)
        loss, stats = self.forward_loss(data, pred, mask, nask, cat_logits=cat_logits)

        if self.training:
            self.global_step += 1

        if return_stats:
            return loss, pred, mask, nask, stats
        return loss, pred, mask, nask
