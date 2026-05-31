# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class FluxTransformer2DArchConfig(DiTArchConfig):
    # Packed latent channels: 16 * 2 * 2 = 64
    in_channels: int = 64
    out_channels: int = 64
    # Double-stream blocks
    num_layers: int = 19
    # Single-stream blocks
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    # T5 sequence dim
    joint_attention_dim: int = 4096
    # CLIP pooled dim
    pooled_projection_dim: int = 768
    # True for FLUX.1-dev, False for FLUX.1-schnell
    guidance_embeds: bool = True
    # RoPE axes: [time/none, height, width]
    axes_dim_rope: list = field(default_factory=lambda: [16, 56, 56])
    patch_size: int = 1
    sample_size: int = 128


@dataclass
class FluxSchnellTransformer2DArchConfig(FluxTransformer2DArchConfig):
    guidance_embeds: bool = False


@dataclass
class FluxDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=FluxTransformer2DArchConfig)
    prefix: str = "flux"


@dataclass
class FluxSchnellDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=FluxSchnellTransformer2DArchConfig)
    prefix: str = "flux_schnell"
