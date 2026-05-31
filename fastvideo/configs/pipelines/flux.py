# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from fastvideo.configs.models import EncoderConfig
from fastvideo.configs.models.encoders import BaseEncoderOutput, CLIPTextConfig, T5Config
from fastvideo.configs.models.dits.flux import FluxDiTConfig, FluxSchnellDiTConfig
from fastvideo.configs.models.vaes.autoencoder_kl import AutoencoderKLVAEConfig
from fastvideo.configs.pipelines.base import PipelineConfig, preprocess_text


def _flux_clip_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    assert outputs.pooler_output is not None, "CLIP pooler_output required for Flux"
    return outputs.pooler_output


def _flux_t5_postprocess(outputs: BaseEncoderOutput) -> torch.Tensor:
    assert outputs.last_hidden_state is not None
    return outputs.last_hidden_state


@dataclass
class FluxConfig(PipelineConfig):
    """FLUX.1-dev pipeline configuration."""

    scheduler_arch: str = "FlowMatchEulerDiscreteScheduler"
    transformer_arch: str = "FluxTransformer2DModel"
    vae_arch: str = "AutoencoderKL"
    text_encoder_archs: tuple[str, ...] = ("CLIPTextModel", "T5EncoderModel")
    tokenizer_archs: tuple[str, ...] = ("CLIPTokenizer", "T5TokenizerFast")

    dit_config: FluxDiTConfig = field(default_factory=FluxDiTConfig)
    vae_config: AutoencoderKLVAEConfig = field(default_factory=AutoencoderKLVAEConfig)

    # Guidance scale used by the dev model's guidance embedding
    guidance_scale: float = 3.5
    flow_shift: float | None = None

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config()))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text, preprocess_text))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = field(
        default_factory=lambda: (_flux_clip_postprocess, _flux_t5_postprocess))

    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32", "bf16"))

    def __post_init__(self) -> None:
        te_cfgs = list(self.text_encoder_configs)
        # CLIP: max_length=77, return pooler
        te_cfgs[0].tokenizer_kwargs.setdefault("padding", "max_length")
        te_cfgs[0].tokenizer_kwargs.setdefault("max_length", 77)
        te_cfgs[0].tokenizer_kwargs.setdefault("truncation", True)
        te_cfgs[0].tokenizer_kwargs.setdefault("return_tensors", "pt")
        # T5: max_length=512
        te_cfgs[1].tokenizer_kwargs.setdefault("padding", "max_length")
        te_cfgs[1].tokenizer_kwargs.setdefault("max_length", 512)
        te_cfgs[1].tokenizer_kwargs.setdefault("truncation", True)
        te_cfgs[1].tokenizer_kwargs.setdefault("return_tensors", "pt")
        self.text_encoder_configs = tuple(te_cfgs)


@dataclass
class FluxSchnellConfig(FluxConfig):
    """FLUX.1-schnell pipeline configuration (distilled, no guidance embedding)."""

    dit_config: FluxSchnellDiTConfig = field(default_factory=FluxSchnellDiTConfig)
    guidance_scale: float = 0.0
