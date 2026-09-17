# Copyright 2025 Ilia Larchenko and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineAnnealingSchedulerConfig


@PreTrainedConfig.register_subclass("dot")
@dataclass
class DOTConfig(PreTrainedConfig):
    """Configuration for the Decoder-Only Transformer (DOT) policy.

    Port of https://github.com/IliaLarchenko/dot_policy to the current LeRobot policy API.

    Parameters that depend on the dataset FPS / task and usually need adjusting:
    - train_horizon: number of future actions predicted during training
    - inference_horizon: number of future actions predicted (and ensembled) at inference
    - lookback_obs_steps / lookback_aug: how far back the single "far past" observation is taken from,
      and the +/- range it is randomly jittered over during training
    - alpha / train_alpha: exponential decay of the action-ensembling and loss weights
    - rescale_shape: all cameras are resized to this before the shared backbone

    Inference speed knobs: predict_every_n (run the model every n steps, shift predictions in between)
    and return_every_n (skip ahead n actions per call).
    """

    # Input / output structure.
    n_obs_steps: int = 3
    train_horizon: int = 20
    inference_horizon: int = 20
    lookback_obs_steps: int = 10
    lookback_aug: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Set False to ignore `observation.environment_state` entirely (no-touch baseline on a touch dataset).
    use_env_state: bool = True
    # Dataset input keys to ignore, e.g. ("observation.images.wrist",) for a single-camera ablation.
    drop_input_features: tuple[str, ...] = ()
    # Environment-state (e.g. tactile array) path. With `env_state_layout` = (sensors, rows, cols) each
    # sensor sheet goes through a small conv stem, is pooled to `env_state_tokens` (h, w) and projected,
    # giving sensors*h*w tokens per observation step instead of one linear token for the flat vector.
    # Auto-filled from the dataset's `observation.environment_state.info.layout` when a single 3-D source exists.
    env_state_layout: tuple[int, int, int] | None = None
    env_state_tokens: tuple[int, int] = (1, 1)
    env_state_stem_channels: int = 32
    # Fixed input scale (sensor counts) applied in the model; ENV normalization stays IDENTITY.
    env_state_scale: float = 512.0

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = True
    lora_rank: int = 20
    merge_lora: bool = False

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (96, 96)

    # Augmentation.
    crop_scale: float = 0.8
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.75
    train_alpha: float = 0.9
    predict_every_n: int = 1
    return_every_n: int = 1
    # Inference-only: run this ONNX export (2026-09-17-dot-policy/torq/export_dot_onnx.py, the cached-embedding
    # graph) through onnxruntime instead of the torch model. Same weights and math, ~3x faster on CPU.
    onnx_path: str | None = None

    # Training preset
    optimizer_lr: float = 1.0e-4
    optimizer_min_lr: float = 1.0e-4
    optimizer_lr_cycle_steps: int = 300000
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 50.0

    def __post_init__(self):
        super().__post_init__()
        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be <= inference_horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be <= inference_horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be <= inference_horizon // return_every_n "
                f"({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be >= inference_horizon ({self.inference_horizon})."
            )
        if self.n_obs_steps < 2:
            raise ValueError(
                f"n_obs_steps must be >= 2 (one lookback + recent steps). Got {self.n_obs_steps}."
            )
        if self.lookback_obs_steps - self.lookback_aug < self.n_obs_steps - 1:
            raise ValueError(
                "lookback_obs_steps - lookback_aug must be >= n_obs_steps - 1 so the far-past window "
                "does not overlap the recent observations."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineAnnealingSchedulerConfig:
        return CosineAnnealingSchedulerConfig(
            min_lr=self.optimizer_min_lr, T_max=self.optimizer_lr_cycle_steps
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    def set_dataset_feature_metadata(self, features: dict) -> None:
        """Apply feature ablations and pick up the env-state layout recorded by `hw_to_dataset_features`."""
        for key in self.drop_input_features:
            if self.input_features.pop(key, None) is None:
                raise ValueError(
                    f"drop_input_features: {key!r} is not an input feature ({list(self.input_features)})"
                )
        if not self.use_env_state:
            self.input_features.pop("observation.environment_state", None)
            self.env_state_layout = None
            return
        if self.env_state_layout is not None:
            return
        layout = ((features.get("observation.environment_state") or {}).get("info") or {}).get("layout") or {}
        if len(layout) == 1:
            shape = next(iter(layout.values()))
            if len(shape) == 3:
                self.env_state_layout = tuple(int(d) for d in shape)

    @property
    def observation_delta_indices(self) -> list:
        far_past_obs = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps
            )
        )
        recent_obs = list(range(2 - self.n_obs_steps, 1))
        return far_past_obs + recent_obs

    @property
    def action_delta_indices(self) -> list:
        far_past_actions = list(
            range(
                -self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps
            )
        )
        recent_actions = list(range(2 - self.n_obs_steps, self.train_horizon))
        return far_past_actions + recent_actions

    @property
    def reward_delta_indices(self) -> None:
        return None
