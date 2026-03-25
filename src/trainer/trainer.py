import argparse
import contextlib
import math
import random
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from config.args import (
    load_config_args,
    merge_dataset_config,
)
from data.preprocessing import (
    TabularDataProcessor,
    infer_feature_type_indices_from_names,
)
from model.maptab import MAPMaskingAutoencoder
from utils.runtime import format_bold
from utils.utils import NativeScaler, adjust_learning_rate

from tqdm import tqdm

EPS = 1e-8


class Trainer:
    def __init__(self, args: Optional[argparse.Namespace] = None):
        if args is None:
            args = load_config_args()

        args = merge_dataset_config(args)

        device_name = args.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
            print(f">> using {device_name}")
        self.device = torch.device(device_name)

        self.batch_size = args.batch_size
        self.accum_iter = args.accum_iter
        self.min_lr = args.min_lr
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.blr = args.blr
        self.warmup_epochs = args.warmup_epochs

        self.embed_dim = args.embed_dim
        self.depth = args.depth
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.mlp_ratio = args.mlp_ratio
        self.max_epochs = args.max_epochs
        self.map_alpha_min = float(getattr(args, "map_alpha_min", 0.1))
        self.map_alpha_max = float(getattr(args, "map_alpha_max", 1.0))
        self.map_gamma = float(getattr(args, "map_gamma", 0.5))
        self.map_weight_a = float(getattr(args, "map_weight_a", 0.5))
        self.map_weight_b = float(getattr(args, "map_weight_b", 0.2))
        self.observed_loss_weight = float(getattr(args, "observed_loss_weight", 1.0))
        self.type_aware_cont_weight = float(
            getattr(args, "type_aware_cont_weight", 0.8)
        )
        self.map_rho_max = float(getattr(args, "map_rho_max", 0.7))
        self.progress_position = int(getattr(args, "progress_position", 0))
        self.progress_desc = str(getattr(args, "progress_desc", "Training"))
        self.progress_disable = bool(getattr(args, "progress_disable", False))
        self.log_disable = bool(getattr(args, "log_disable", False))

        self.model = None
        self.norm_parameters = None
        self.training_history = []

        self.data_processor = TabularDataProcessor()

    @staticmethod
    def name() -> str:
        return "ours"

    def _log(self, message: str) -> None:
        if not self.log_disable:
            print(message)

    def _validate_dataframe_schema(self, x_raw: pd.DataFrame) -> None:
        _, _, unknown = infer_feature_type_indices_from_names(x_raw.columns)
        if unknown:
            unknown_cols = [str(x_raw.columns[idx]) for idx in unknown]
            raise ValueError(
                "All feature columns must start with 'num' or 'cat'. "
                f"Columns without valid prefix: {unknown_cols}"
            )

    @staticmethod
    def set_seed(seed: int):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self, x_raw):

        if not isinstance(x_raw, pd.DataFrame):
            raise ValueError(
                "fit(...) requires a pandas DataFrame so the code can enforce the 'num*'/'cat*' schema."
            )
        self._validate_dataframe_schema(x_raw)
        self.data_processor.set_feature_names(list(x_raw.columns))
        self._log(
            f"[Trainer] fit start | raw_dataframe_shape={x_raw.shape} | device={self.device}"
        )
        x_np = x_raw.to_numpy(dtype=np.float32)
        x_norm, m_np = self.data_processor.fit_transform(x_np)
        self.norm_parameters = self.data_processor.norm_parameters
        self._log("[Trainer] normalization completed")
        self.data_processor.log_feature_type_summary(self._log)
        self.training_history = []

        x = torch.from_numpy(x_norm).to(self.device)
        m = torch.from_numpy(m_np).to(self.device)

        dim = x.shape[1]
        model_kwargs = dict(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=EPS),
        )

        rho_max = self.map_rho_max
        categorical_feature_mask = torch.from_numpy(
            self.norm_parameters["categorical_feature_mask"].astype(bool)
        )
        categorical_cardinalities = self.norm_parameters["cat_cardinalities"]
        self.model = MAPMaskingAutoencoder(
            alpha_min=self.map_alpha_min,
            alpha_max=self.map_alpha_max,
            gamma=self.map_gamma,
            rho_max=rho_max,
            weight_a=self.map_weight_a,
            weight_b=self.map_weight_b,
            observed_loss_weight=self.observed_loss_weight,
            type_aware_cont_weight=self.type_aware_cont_weight,
            total_steps=max(int(self.max_epochs), 1),
            categorical_feature_mask=categorical_feature_mask,
            categorical_cardinalities=categorical_cardinalities,
            **model_kwargs,
        ).to(self.device)
        remaskable_feature_mask = torch.ones(
            m.shape[1], dtype=torch.bool, device=m.device
        )
        global_missing_rate = (1.0 - m.float().mean(dim=0)).detach()
        self.model.set_remaskable_features(remaskable_feature_mask)
        self.model.set_global_missing_rate(global_missing_rate)
        self._log(
            f"[Trainer] model initialized | model=maptab | rec_len={dim} | "
            f"embed_dim={self.embed_dim} | depth={self.depth} | decoder_depth={self.decoder_depth} | "
            f"heads={self.num_heads}"
        )
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # Automatically scale the learning rate with batch size.
            self.lr = self.blr * eff_batch_size / 64

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )
        loss_scaler = NativeScaler(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        )

        num_samples = int(x.shape[0])
        steps_per_epoch = max(int(math.ceil(num_samples / float(self.batch_size))), 1)
        self._log(
            f"[Trainer] dataloader ready | samples={num_samples} | batch_size={format_bold(self.batch_size)} | "
            f"steps_per_epoch={steps_per_epoch} | max_epochs={self.max_epochs}"
        )
        total_steps = self.max_epochs * steps_per_epoch
        self.model.set_total_steps(max(int(total_steps), 1))
        self.model.reset_progress()
        self.model.train()
        amp_context = (
            partial(torch.amp.autocast, self.device.type)
            if self.device.type == "cuda"
            else contextlib.nullcontext
        )

        epoch_iter = tqdm(
            range(self.max_epochs),
            desc=self.progress_desc,
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
            position=max(0, self.progress_position),
            disable=self.progress_disable,
        )

        for epoch in epoch_iter:
            optimizer.zero_grad()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            dyn_steps = 0
            dyn_alpha_last = 0.0

            permutation = torch.randperm(num_samples, device=x.device)
            for it in range(steps_per_epoch):
                batch_idx = permutation[
                    it * self.batch_size : min((it + 1) * self.batch_size, num_samples)
                ]
                if it % self.accum_iter == 0:
                    # Update the learning rate per iteration to stay compatible with gradient accumulation.
                    adjust_learning_rate(
                        optimizer,
                        it / steps_per_epoch + epoch,
                        self.lr,
                        self.min_lr,
                        self.max_epochs,
                        self.warmup_epochs,
                    )

                samples = x.index_select(0, batch_idx).unsqueeze(dim=1)
                masks = m.index_select(0, batch_idx)

                with amp_context():
                    loss, _, re_mask, _ = self.model(samples, masks)

                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"Non-finite loss detected during training: {loss_value}"
                    )

                dyn_steps += 1
                dyn_alpha_last = float(getattr(self.model, "last_alpha_t", 0.0))

                epoch_loss_sum += loss_value
                epoch_steps += 1

                loss = loss / self.accum_iter
                should_update = (it + 1) % self.accum_iter == 0 or (
                    it + 1
                ) == steps_per_epoch
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=self.model.parameters(),
                    update_grad=should_update,
                )

                if should_update:
                    optimizer.zero_grad()

            mean_loss = epoch_loss_sum / max(epoch_steps, 1)

            postfix = {
                "loss": f"{mean_loss:.4f}",
            }
            if dyn_steps > 0:
                postfix["alpha"] = f"{dyn_alpha_last:.2f}"
            epoch_iter.set_postfix(**postfix)
            epoch_record = {
                "epoch": int(epoch + 1),
                "loss": float(mean_loss),
                "alpha": float(dyn_alpha_last) if dyn_steps > 0 else None,
            }
            self.training_history.append(epoch_record)

        self._log("[Trainer] fit completed")
        return self

    def transform(self, x_raw) -> torch.Tensor:

        if self.model is None or self.data_processor.norm_parameters is None:
            raise RuntimeError("Trainer is not fitted. Call fit(...) first.")
        if not isinstance(x_raw, pd.DataFrame):
            raise ValueError(
                "transform(...) requires a pandas DataFrame so the code can enforce the 'num*'/'cat*' schema."
            )
        self._validate_dataframe_schema(x_raw)

        self.data_processor.set_feature_names(list(x_raw.columns))
        self._log(f"[Trainer] transform start | raw_dataframe_shape={x_raw.shape}")
        x_np = x_raw.to_numpy(dtype=np.float32)
        x_norm, m_np = self.data_processor.transform(x_np)

        x = torch.from_numpy(x_norm).to(self.device)
        m = torch.from_numpy(m_np).to(self.device)

        self.model.eval()
        preds = []
        infer_batch_size = max(int(self.batch_size or 1), 1)

        with torch.no_grad():

            for start in range(0, x.shape[0], infer_batch_size):
                end = min(start + infer_batch_size, x.shape[0])
                samples = x[start:end].unsqueeze(dim=1)
                masks = m[start:end]
                pred, _ = self.model.reconstruct(samples, masks)
                preds.append(pred.squeeze(2).detach().cpu().numpy())

        imputed_proc = np.concatenate(preds, axis=0)
        imputed = self.data_processor.decode_predictions(imputed_proc)

        if np.isnan(imputed).all():
            raise RuntimeError("The imputed result contains only NaN values.")

        observed = np.nan_to_num(x_np, nan=0.0)
        merged = m_np * observed + (1.0 - m_np) * imputed
        self._log(f"[Trainer] transform completed | output_shape={merged.shape}")
        return torch.from_numpy(merged.astype(np.float32))

    def fit_transform(self, x):
        return self.fit(x).transform(x).detach().cpu().numpy()

    def get_training_history(self):
        return list(self.training_history)
