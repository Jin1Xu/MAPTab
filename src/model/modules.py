import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from utils.utils import (
    MixedFeatureEmbed,
    get_1d_sincos_pos_embed,
)


def configure_feature_types(
    model,
    rec_len: int,
    categorical_feature_mask=None,
    categorical_cardinalities=None,
) -> None:
    if categorical_feature_mask is None:
        categorical_feature_mask = torch.zeros(rec_len, dtype=torch.bool)
    else:
        categorical_feature_mask = torch.as_tensor(
            categorical_feature_mask, dtype=torch.bool
        )
    if categorical_feature_mask.numel() != rec_len:
        raise ValueError(
            f"categorical_feature_mask length mismatch: expected {rec_len}, got {categorical_feature_mask.numel()}"
        )

    if categorical_cardinalities is None:
        categorical_cardinalities = [
            1 if bool(is_cat) else 0 for is_cat in categorical_feature_mask.tolist()
        ]
    if len(categorical_cardinalities) != rec_len:
        raise ValueError(
            f"categorical_cardinalities length mismatch: expected {rec_len}, got {len(categorical_cardinalities)}"
        )
    categorical_cardinalities = [
        max(int(x), 1) if bool(categorical_feature_mask[i]) else 0
        for i, x in enumerate(categorical_cardinalities)
    ]

    model.register_buffer(
        "categorical_feature_mask", categorical_feature_mask, persistent=True
    )
    model.register_buffer(
        "continuous_feature_mask", ~categorical_feature_mask, persistent=True
    )
    model.categorical_cardinalities = categorical_cardinalities
    model.categorical_indices = (
        torch.nonzero(model.categorical_feature_mask, as_tuple=False)
        .squeeze(1)
        .tolist()
    )
    model.continuous_indices = (
        torch.nonzero(model.continuous_feature_mask, as_tuple=False).squeeze(1).tolist()
    )


def build_model_components(
    model,
    rec_len: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    decoder_embed_dim: int,
    decoder_depth: int,
    decoder_num_heads: int,
    mlp_ratio: float,
    norm_layer,
) -> None:
    model.mask_embed = MixedFeatureEmbed(
        rec_len=rec_len,
        embed_dim=embed_dim,
        categorical_feature_mask=model.categorical_feature_mask,
        categorical_cardinalities=model.categorical_cardinalities,
    )

    model.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    model.pos_embed = nn.Parameter(
        torch.zeros(1, rec_len + 1, embed_dim), requires_grad=False
    )
    model.col_embed = nn.Parameter(
        torch.zeros(1, rec_len + 1, embed_dim), requires_grad=True
    )

    model.encoder_blocks = nn.ModuleList(
        [
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ]
    )
    model.norm = norm_layer(embed_dim)

    model.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
    model.decoder_pos_embed = nn.Parameter(
        torch.zeros(1, rec_len + 1, decoder_embed_dim), requires_grad=False
    )
    model.decoder_blocks = nn.ModuleList(
        [
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ]
    )
    model.decoder_norm = norm_layer(decoder_embed_dim)
    model.decoder_pred_cont = nn.Linear(decoder_embed_dim, 1, bias=True)
    model.decoder_pred_cat = nn.ModuleDict(
        {
            str(idx): nn.Linear(
                decoder_embed_dim,
                model.categorical_cardinalities[idx],
                bias=True,
            )
            for idx in model.categorical_indices
        }
    )

    model.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))


def _init_weights(module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def initialize_model_weights(model) -> None:
    pos_embed = get_1d_sincos_pos_embed(
        model.pos_embed.shape[-1], model.mask_embed.rec_len, cls_token=True
    )
    model.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    decoder_pos_embed = get_1d_sincos_pos_embed(
        model.decoder_pos_embed.shape[-1], model.mask_embed.rec_len, cls_token=True
    )
    model.decoder_pos_embed.data.copy_(
        torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
    )

    torch.nn.init.xavier_uniform_(model.mask_embed.con_proj.weight)
    if model.mask_embed.con_proj.bias is not None:
        nn.init.constant_(model.mask_embed.con_proj.bias, 0)
    for emb in model.mask_embed.cat_embeddings.values():
        torch.nn.init.normal_(emb.weight, std=0.02)

    torch.nn.init.normal_(model.cls_token, std=0.02)
    torch.nn.init.normal_(model.encoder_mask_token, std=0.02)
    torch.nn.init.normal_(model.col_embed, std=0.02)

    model.apply(_init_weights)


def run_encoder(model, x: torch.Tensor, m: torch.Tensor):
    x = model.mask_embed(x)
    x = x + model.pos_embed[:, 1:, :]
    x = x + model.col_embed[:, 1:, :]

    x, mask, nask, encoder_mask_pos = model.map_masking(x, m)

    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_token = cls_token + model.col_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    key_padding_mask = torch.cat(
        [
            torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device),
            encoder_mask_pos,
        ],
        dim=1,
    )
    for blk in model.encoder_blocks:
        x = blk(x, src_key_padding_mask=key_padding_mask)

    return model.norm(x), mask, nask


def run_decoder(model, x: torch.Tensor):
    x = model.decoder_embed(x)
    x = x + model.decoder_pos_embed

    for blk in model.decoder_blocks:
        x = blk(x)

    x = model.decoder_norm(x)
    token_states = x[:, 1:, :]

    pred = torch.zeros(
        token_states.shape[0], token_states.shape[1], 1, device=token_states.device
    )
    cat_logits = {}

    for idx in model.continuous_indices:
        pred[:, idx : idx + 1, :] = (
            torch.tanh(model.decoder_pred_cont(token_states[:, idx : idx + 1, :])) / 2
            + 0.5
        )
    for idx in model.categorical_indices:
        logits = model.decoder_pred_cat[str(idx)](token_states[:, idx, :])
        cat_logits[idx] = logits
        pred[:, idx, 0] = torch.argmax(logits, dim=1).float()

    return pred, cat_logits
