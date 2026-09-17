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
"""Decoder-Only Transformer (DOT) policy.

Port of https://github.com/IliaLarchenko/dot_policy: every input (states, images, env state) is
projected to `dim_model` and fed as memory to a plain `nn.TransformerDecoder` whose queries are
sinusoidal positional encodings for the action horizon. Non-generative, L1 loss, LoRA on the ResNet.
"""

import math

import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import InterpolationMode

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_dot import DOTConfig


class DOT(nn.Module):
    def __init__(self, config: DOTConfig):
        super().__init__()
        self.config = config

        self.projections = nn.ModuleDict()
        self.n_features = 0

        self.image_names = sorted(config.image_features.keys())

        # One backbone for all cameras; its fc layer is the projection to dim_model.
        if len(self.image_names) > 0:
            backbone = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            backbone.fc = nn.Linear(backbone.fc.in_features, config.dim_model)
            self.projections["images"] = add_lora_to_backbone(backbone, rank=config.lora_rank)
            self.n_features += len(self.image_names) * config.n_obs_steps

        if config.robot_state_feature:
            self.projections["state"] = nn.Linear(config.robot_state_feature.shape[0], config.dim_model)
            self.n_features += config.n_obs_steps

        if config.env_state_feature:
            if config.env_state_layout:
                stem = DOTEnvStateStem(config)
                self.projections["environment_state"] = stem
                self.n_features += config.n_obs_steps * stem.n_tokens
            else:
                self.projections["environment_state"] = nn.Sequential(
                    ScaleInput(config.env_state_scale),
                    nn.Linear(config.env_state_feature.shape[0], config.dim_model),
                )
                self.n_features += config.n_obs_steps

        self.projections_names = sorted(self.projections.keys())
        obs_mapping = {
            "images": OBS_IMAGES,
            "state": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
        }
        self.obs_mapping = {k: v for k, v in obs_mapping.items() if k in self.projections_names}

        # Extra trainable memory token.
        self.prefix_input = nn.Parameter(torch.randn(1, 1, config.dim_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=config.pre_norm,
        )
        decoder_norm = nn.LayerNorm(config.dim_model)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.n_decoder_layers, norm=decoder_norm)

        # Decoder queries: fixed sinusoidal encodings, one per predicted step (lookback step + recent + horizon).
        decoder_pos = create_sinusoidal_pos_embedding(
            config.train_horizon + config.lookback_obs_steps, config.dim_model
        )
        decoder_pos = torch.cat(
            [decoder_pos[:1], decoder_pos[-config.train_horizon - config.n_obs_steps + 2 :]],
            dim=0,
        )
        self.register_buffer("decoder_pos", decoder_pos)

        decoder_pos_inf = self.decoder_pos[
            : self.decoder_pos.shape[0] + config.inference_horizon - config.train_horizon
        ]
        self.register_buffer("decoder_pos_inf", decoder_pos_inf)

        # Training mask: the first inference_horizon queries cannot attend to the extra training-only steps,
        # so their behaviour matches inference exactly.
        mask = torch.zeros(len(decoder_pos), len(decoder_pos), dtype=torch.bool)
        mask[
            : len(decoder_pos) + config.inference_horizon - config.train_horizon,
            len(decoder_pos) + config.inference_horizon - config.train_horizon :,
        ] = True
        self.register_buffer("mask", mask)

        # Memory tokens get trainable positional embeddings.
        self.inputs_pos_emb = nn.Parameter(torch.empty(1, self.n_features, config.dim_model))
        nn.init.uniform_(self.inputs_pos_emb, -((1 / config.dim_model) ** 0.5), (1 / config.dim_model) ** 0.5)

        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

    def _process_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        """Project every input to dim_model and concatenate along the token axis.

        ``batch["_projections"]`` may hold already-projected inputs, ``{name: (B, n_obs, tokens, dim)}``,
        which are used in place of running that projection (the inference-time image feature cache).
        """
        precomputed = batch.get("_projections", {})
        inputs_projections_list = []
        for state in self.projections_names:
            batch_state = self.obs_mapping[state]
            if state in precomputed:
                inputs_projections_list.append(precomputed[state].flatten(1, 2))
            elif batch_state in batch:
                bs, n_obs, *obs_shape = batch[batch_state].shape
                enc = self.projections[state](batch[batch_state].reshape(bs * n_obs, *obs_shape)).reshape(
                    bs, n_obs, -1, self.config.dim_model
                )
                inputs_projections_list.append(enc.flatten(1, 2))
        return torch.cat(inputs_projections_list, dim=1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        inputs_projections = self._process_inputs(batch)
        bs = inputs_projections.shape[0]

        inputs_projections = inputs_projections + self.inputs_pos_emb.expand(bs, -1, -1)
        inputs_projections = torch.cat([self.prefix_input.expand(bs, -1, -1), inputs_projections], dim=1)

        if self.training:
            decoder_out = self.decoder(self.decoder_pos.expand(bs, -1, -1), inputs_projections, self.mask)
        else:
            decoder_out = self.decoder(self.decoder_pos_inf.expand(bs, -1, -1), inputs_projections)
        return self.action_head(decoder_out)


class DOTPolicy(PreTrainedPolicy):
    """Decoder-Only Transformer policy (https://github.com/IliaLarchenko/dot_policy)."""

    config_class = DOTConfig
    name = "dot"

    def __init__(self, config: DOTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.image_names = sorted(config.image_features.keys())
        self.model = DOT(config)

        # Augmentation strength decays over training (not checkpointed, same as upstream).
        self.state_noise = config.state_noise
        self.crop_scale = config.crop_scale

        # Exponential weights for ensembling overlapping inference chunks.
        action_weights = config.alpha ** torch.arange(config.inference_horizon).float()
        action_weights /= action_weights.sum()
        self.register_buffer("action_weights", action_weights.view(1, -1, 1))

        # Loss weights: far-future actions count less.
        loss_weights = torch.ones(config.train_horizon + config.n_obs_steps - 1)
        loss_weights[-config.train_horizon :] = (
            config.train_alpha ** torch.arange(config.train_horizon).float()
        )
        loss_weights /= loss_weights.mean()
        self.register_buffer("loss_weights", loss_weights.view(1, -1, 1))

        self.resize_transform = transforms.Resize(
            config.rescale_shape, interpolation=InterpolationMode.NEAREST
        )

        self.reset()

    def reset(self):
        self._old_predictions = None
        self._input_buffers = {}
        # Per-frame backbone features of the image buffer, (B, lookback_obs_steps + 1, n_cam, dim_model),
        # and which of those slots are up to date. Filled lazily: a frame is encoded at most once.
        self._image_features = None
        self._image_features_valid = None
        self.last_action = None
        self.step = 0

    def get_optim_params(self) -> dict:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _update_observation_buffers(self, buffer_name: str, observation: Tensor) -> Tensor:
        """Keep the last lookback_obs_steps + 1 observations; return [lookback, recent n_obs_steps - 1]."""
        if buffer_name not in self._input_buffers:
            self._input_buffers[buffer_name] = observation.unsqueeze(1).repeat(
                1, self.config.lookback_obs_steps + 1, *([1] * (observation.ndim - 1))
            )
        else:
            self._input_buffers[buffer_name] = self._input_buffers[buffer_name].roll(shifts=-1, dims=1)
            self._input_buffers[buffer_name][:, -1] = observation
            if buffer_name == "images" and self._image_features_valid is not None:
                self._image_features = self._image_features.roll(shifts=-1, dims=1)
                self._image_features_valid = self._image_features_valid.roll(shifts=-1, dims=0)
                self._image_features_valid[-1] = False

        return torch.cat(
            [
                self._input_buffers[buffer_name][:, :1],
                self._input_buffers[buffer_name][:, -(self.config.n_obs_steps - 1) :],
            ],
            dim=1,
        )

    def _prepare_batch_for_inference(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy: we add keys
        if len(self.image_names) > 0:
            batch[OBS_IMAGES] = torch.stack(
                [self.resize_transform(batch[k]) for k in self.image_names], dim=1
            )
            # (B, n_cam, C, H, W)

        for name, batch_name in self.model.obs_mapping.items():
            batch[batch_name] = self._update_observation_buffers(name, batch[batch_name])

        if OBS_IMAGES in batch:
            batch[OBS_IMAGES] = batch[OBS_IMAGES].flatten(
                1, 2
            )  # (B, n_obs * n_cam, C, H, W), same order as training
        return batch

    def _cached_image_projections(self) -> Tensor:
        """Backbone features for the [lookback, recent n_obs_steps - 1] image frames, (B, n_obs * n_cam, 1, dim).

        Every frame in the raw image buffer was already seen on an earlier tick, so its features are
        cached and only the slots that changed since (normally just the newest frame) go through the
        backbone. Output matches projecting the raw frames directly, just without the redundant passes.
        """
        raw = self._input_buffers["images"]  # (B, lookback + 1, n_cam, C, H, W)
        bs, n_slots, n_cam = raw.shape[:3]
        proj = self.model.projections["images"]
        if self._image_features_valid is None:
            self._image_features = raw.new_zeros(bs, n_slots, n_cam, self.config.dim_model)
            self._image_features_valid = torch.zeros(n_slots, dtype=torch.bool, device=raw.device)

        needed = [0] + list(range(n_slots - (self.config.n_obs_steps - 1), n_slots))
        stale = [i for i in needed if not self._image_features_valid[i]]
        if stale:
            frames = raw[:, stale].flatten(0, 2)  # (B * len(stale) * n_cam, C, H, W)
            feats = proj(frames).reshape(bs, len(stale), n_cam, -1)
            self._image_features[:, stale] = feats
            self._image_features_valid[stale] = True

        return self._image_features[:, needed].flatten(1, 2).unsqueeze(2)

    def _chunk_actions(self, actions: Tensor) -> Tensor:
        """Exponentially weighted average of the overlapping predictions for the current step."""
        if self._old_predictions is not None:
            self._old_predictions[:, 0] = actions
        else:
            self._old_predictions = actions.unsqueeze(1).repeat(1, self.config.inference_horizon, 1, 1)

        action = (self._old_predictions[:, :, 0] * self.action_weights).sum(dim=1)
        self._old_predictions = self._old_predictions.roll(shifts=(1, -1), dims=(1, 2))
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict `inference_horizon` (normalized) actions; also advances the observation buffers."""
        self.eval()
        batch = self._prepare_batch_for_inference(batch)
        if OBS_IMAGES in batch:
            batch["_projections"] = {"images": self._cached_image_projections()}
        return self.model(batch)[:, -self.config.inference_horizon :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        if self.step % self.config.predict_every_n == 0:
            self.last_action = self.predict_action_chunk(batch)
        else:
            # Advance the buffers anyway so the lookback history stays in sync with time.
            self._prepare_batch_for_inference(batch)
            self.last_action = self.last_action.roll(-1, dims=1)
            self.last_action[:, -1] = self.last_action[:, -2]

        self.step += 1

        action = self._chunk_actions(self.last_action)
        for _ in range(self.config.return_every_n - 1):
            self.last_action = self.last_action.roll(-1, dims=1)
            self.last_action[:, -1] = self.last_action[:, -2]
            action = self._chunk_actions(self.last_action)
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training loss. Inputs arrive already normalized by the preprocessor.

        Observation keys hold (B, 2*lookback_aug+1 + n_obs_steps-1, ...) frames: the far-past window
        followed by the recent steps. One far-past frame is picked at random per batch (lookback aug).
        """
        batch = dict(batch)
        lookback_ind = torch.randint(0, 2 * self.config.lookback_aug + 1, (1,)).item()
        for k in list(self.model.obs_mapping.values()) + list(self.image_names) + [ACTION, "action_is_pad"]:
            if k != OBS_IMAGES:
                batch[k] = torch.cat(
                    [
                        batch[k][:, lookback_ind : lookback_ind + 1],
                        batch[k][:, 2 * self.config.lookback_aug + 1 :],
                    ],
                    dim=1,
                )

        if len(self.image_names) > 0:
            # One random crop scale per batch, then a random crop of the resized images.
            scale = 1 - torch.rand(1).item() * (1 - self.crop_scale)
            new_shape = (int(self.config.rescale_shape[0] * scale), int(self.config.rescale_shape[1] * scale))
            crop_transform = transforms.RandomCrop(new_shape)

            for k in self.image_names:
                bs, n_obs, c, h, w = batch[k].shape
                imgs = crop_transform(self.resize_transform(batch[k].reshape(bs * n_obs, c, h, w)))
                batch[k] = imgs.reshape(bs, n_obs, c, *imgs.shape[-2:])
            batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.image_names], dim=2).flatten(1, 2)
            # (B, n_obs * n_cam, C, h, w)

        # Uniform noise on the (normalized) robot state; env state is scaled inside its projection,
        # so its noise is expressed in the same units.
        if self.state_noise:
            if OBS_STATE in batch:
                batch[OBS_STATE] = (
                    batch[OBS_STATE] + (torch.rand_like(batch[OBS_STATE]) * 2 - 1) * self.state_noise
                )
            if OBS_ENV_STATE in batch:
                noise = (
                    (torch.rand_like(batch[OBS_ENV_STATE]) * 2 - 1)
                    * self.state_noise
                    * self.config.env_state_scale
                )
                batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE] + noise

        actions_hat = self.model(batch)

        loss = nn.functional.l1_loss(batch[ACTION], actions_hat, reduction="none")
        rev_padding = (~batch["action_is_pad"]).unsqueeze(-1)
        loss = loss * rev_padding * self.loss_weights

        if self.training:
            # Anneal the augmentations.
            self.state_noise *= self.config.noise_decay
            self.crop_scale = 1 - (1 - self.crop_scale) * self.config.noise_decay

        if reduction == "none":
            return loss.mean(dim=(1, 2)), {}
        loss = loss.mean()
        return loss, {"l1_loss": loss.item()}

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Load, then optionally fold the LoRA updates into the backbone convolutions."""
        policy = super().from_pretrained(pretrained_name_or_path, *args, **kwargs)
        if getattr(policy.config, "merge_lora", False):
            policy.model = merge_lora_weights(policy.model)
        return policy


class ScaleInput(nn.Module):
    """Divide by a fixed constant (env state is not normalized by the preprocessor)."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        return x / self.scale


class DOTEnvStateStem(nn.Module):
    """Conv stem for a grid-shaped environment state such as a tactile array.

    Input (N, S*H*W) laid out as `env_state_layout` = (S, H, W). Each sensor sheet is scaled by a constant,
    passed through two 3x3 convs (shared across sensors), pooled to `env_state_tokens` (h, w) and projected
    to dim_model, giving S*h*w tokens per observation. Output (N, S*h*w, dim_model).
    """

    def __init__(self, config: DOTConfig):
        super().__init__()
        self.layout = tuple(config.env_state_layout)
        self.scale = float(config.env_state_scale)
        ch = config.env_state_stem_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(tuple(config.env_state_tokens))
        self.proj = nn.Conv2d(ch, config.dim_model, kernel_size=1)
        self.n_tokens = self.layout[0] * config.env_state_tokens[0] * config.env_state_tokens[1]

    def forward(self, x: Tensor) -> Tensor:
        n_sensors, rows, cols = self.layout
        n = x.shape[0]
        sheets = x.reshape(n * n_sensors, 1, rows, cols) / self.scale
        feat = self.proj(self.pool(self.conv(sheets)))  # (N*S, D, h, w)
        return feat.reshape(n, n_sensors, feat.shape[1], -1).permute(0, 1, 3, 2).reshape(n, self.n_tokens, -1)


class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.base_conv = base_conv
        out_channels, in_channels, kh, kw = base_conv.weight.shape
        self.weight_shape = (out_channels, in_channels, kh, kw)
        fan_in = in_channels * kh * kw
        self.lora_A = nn.Parameter(torch.normal(0, 0.02, (out_channels, rank)))
        self.lora_B = nn.Parameter(torch.normal(0, 0.02, (rank, fan_in)))

    def forward(self, x: Tensor) -> Tensor:
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        return nn.functional.conv2d(
            x,
            self.base_conv.weight + lora_update,
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )

    @torch.no_grad()
    def merge_lora(self) -> nn.Conv2d:
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        self.base_conv.weight.copy_(self.base_conv.weight + lora_update)
        return self.base_conv


def replace_conv2d_with_lora(module: nn.Module, rank: int = 4) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank))
        else:
            replace_conv2d_with_lora(child, rank)
    return module


def merge_lora_weights(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(module, name, child.merge_lora())
        else:
            merge_lora_weights(child)
    return module


def add_lora_to_backbone(backbone: nn.Module, rank: int = 4) -> nn.Module:
    """Freeze the backbone; train only the LoRA factors and the final fc projection."""
    replace_conv2d_with_lora(backbone, rank)
    for name, param in backbone.named_parameters():
        param.requires_grad = "lora_" in name or name.startswith("fc")
    return backbone


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    position = torch.arange(num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float) * (-math.log(10000.0) / dimension))
    pe = torch.zeros(num_positions, dimension)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
