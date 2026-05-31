# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models import DiTConfig
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.float()


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # xq/xk: (B, H, S, D)  freqs: (1, 1, S, D/2, 2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    """Multi-dimensional RoPE frequency embeddings."""

    def __init__(self, dim: int, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


# ---------------------------------------------------------------------------
# Normalisation layers
# ---------------------------------------------------------------------------

class FluxRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.scale, eps=self.eps)


class FluxAdaLayerNormZero(nn.Module):
    """AdaLN-Zero used in double-stream blocks (6 modulation params)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple:
        out = self.linear(F.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = out.chunk(6, dim=1)
        normed = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return normed, gate_msa, shift_mlp, scale_mlp, gate_mlp


class FluxAdaLayerNormZeroSingle(nn.Module):
    """AdaLN-Zero used in single-stream blocks (3 modulation params)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 3 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple:
        out = self.linear(F.silu(emb))
        shift_msa, scale_msa, gate_msa = out.chunk(3, dim=1)
        normed = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return normed, gate_msa


class FluxAdaLayerNormContinuous(nn.Module):
    """Output norm conditioned on the timestep embedding."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.linear(F.silu(cond))
        scale, shift = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


# ---------------------------------------------------------------------------
# Timestep / guidance embeddings
# ---------------------------------------------------------------------------

class Timesteps(nn.Module):

    def __init__(self, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.time_proj = Timesteps(frequency_embedding_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.time_proj(t))


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Timestep + pooled CLIP text embedding (FLUX.1-schnell, no guidance)."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedder(embedding_dim)
        self.text_embedder = nn.Sequential(
            nn.Linear(pooled_projection_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(timestep) + self.text_embedder(pooled_projection)


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Timestep + guidance + pooled CLIP text embedding (FLUX.1-dev)."""

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(256)
        self.timestep_embedder = TimestepEmbedder(embedding_dim)
        self.guidance_embedder = TimestepEmbedder(embedding_dim)
        self.text_embedder = nn.Sequential(
            nn.Linear(pooled_projection_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.timestep_embedder(timestep) + self.text_embedder(pooled_projection)
        if guidance is not None:
            temb = temb + self.guidance_embedder(guidance)
        return temb


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class FluxAttention(nn.Module):
    """Joint image+text attention used in double-stream blocks."""

    def __init__(self, query_dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        inner = heads * dim_head
        self.to_q = nn.Linear(query_dim, inner, bias=True)
        self.to_k = nn.Linear(query_dim, inner, bias=True)
        self.to_v = nn.Linear(query_dim, inner, bias=True)
        self.to_out = nn.Linear(inner, query_dim, bias=True)

        self.add_q_proj = nn.Linear(query_dim, inner, bias=True)
        self.add_k_proj = nn.Linear(query_dim, inner, bias=True)
        self.add_v_proj = nn.Linear(query_dim, inner, bias=True)
        self.to_add_out = nn.Linear(inner, query_dim, bias=True)

        self.norm_q = FluxRMSNorm(dim_head)
        self.norm_k = FluxRMSNorm(dim_head)
        self.norm_added_q = FluxRMSNorm(dim_head)
        self.norm_added_k = FluxRMSNorm(dim_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = hidden_states.shape[0]
        H = self.heads

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, H, self.dim_head).transpose(1, 2)

        img_q = self.norm_q(reshape(self.to_q(hidden_states)))
        img_k = self.norm_k(reshape(self.to_k(hidden_states)))
        img_v = reshape(self.to_v(hidden_states))

        txt_q = self.norm_added_q(reshape(self.add_q_proj(encoder_hidden_states)))
        txt_k = self.norm_added_k(reshape(self.add_k_proj(encoder_hidden_states)))
        txt_v = reshape(self.add_v_proj(encoder_hidden_states))

        # Concatenate text + image for joint attention
        q = torch.cat([txt_q, img_q], dim=2)
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Apply RoPE to the full joint sequence
        q, k = apply_rope(q, k, image_rotary_emb)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, -1, H * self.dim_head)

        txt_len = encoder_hidden_states.shape[1]
        txt_out = self.to_add_out(out[:, :txt_len])
        img_out = self.to_out(out[:, txt_len:])
        return img_out, txt_out


class FluxSingleAttention(nn.Module):
    """Self-attention used in single-stream blocks."""

    def __init__(self, query_dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner = heads * dim_head

        self.to_q = nn.Linear(query_dim, inner, bias=True)
        self.to_k = nn.Linear(query_dim, inner, bias=True)
        self.to_v = nn.Linear(query_dim, inner, bias=True)
        self.to_out = nn.Linear(inner, query_dim, bias=True)

        self.norm_q = FluxRMSNorm(dim_head)
        self.norm_k = FluxRMSNorm(dim_head)

    def forward(self, hidden_states: torch.Tensor, image_rotary_emb: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        H = self.heads

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, H, self.dim_head).transpose(1, 2)

        q = self.norm_q(reshape(self.to_q(hidden_states)))
        k = self.norm_k(reshape(self.to_k(hidden_states)))
        v = reshape(self.to_v(hidden_states))

        q, k = apply_rope(q, k, image_rotary_emb)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, S, H * self.dim_head)
        out, _ = self.to_out(out) if isinstance(self.to_out, ReplicatedLinear) else (self.to_out(out), None)
        return out


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class FluxTransformerBlock(nn.Module):
    """Double-stream block: separate image and text streams with joint attn."""

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_dim: int) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * 4)

        self.norm1 = FluxAdaLayerNormZero(hidden_size)
        self.norm1_context = FluxAdaLayerNormZero(hidden_size)

        self.attn = FluxAttention(hidden_size, num_attention_heads, attention_head_dim)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        self.ff_context = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_img, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        norm_txt, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)

        img_attn, txt_attn = self.attn(norm_img, norm_txt, image_rotary_emb)

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * img_attn
        norm_img2 = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_img2)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * txt_attn
        norm_txt2 = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_txt2)

        return hidden_states, encoder_hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream block: merged image+text with combined attn+MLP."""

    def __init__(self, hidden_size: int, num_attention_heads: int, attention_head_dim: int,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.mlp_hidden = int(hidden_size * mlp_ratio)

        self.norm = FluxAdaLayerNormZeroSingle(hidden_size)
        self.proj_mlp = nn.Linear(hidden_size, self.mlp_hidden, bias=True)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + self.mlp_hidden, hidden_size, bias=True)

        self.attn = FluxSingleAttention(hidden_size, num_attention_heads, attention_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_hidden, gate = self.norm(hidden_states, temb)

        mlp_out = self.act_mlp(self.proj_mlp(norm_hidden))
        attn_out = self.attn(norm_hidden, image_rotary_emb)

        combined = torch.cat([attn_out, mlp_out], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(combined)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@dataclass
class FluxTransformer2DModelOutput:
    sample: torch.Tensor


class FluxTransformer2DModel(BaseDiT):
    """FastVideo-native FLUX.1 Transformer2DModel.

    Supports both FLUX.1-dev (with guidance embedding) and
    FLUX.1-schnell (without guidance embedding).
    Weight names match the diffusers FluxTransformer2DModel checkpoint layout
    so checkpoints load without remapping.
    """

    _fsdp_shard_conditions = [
        lambda n, m: "transformer_blocks" in n and n.split(".")[-1].isdigit(),
        lambda n, m: "single_transformer_blocks" in n and n.split(".")[-1].isdigit(),
    ]
    _compile_conditions = _fsdp_shard_conditions
    param_names_mapping: dict[str, Any] = {}
    reverse_param_names_mapping: dict[str, Any] = {}
    lora_param_names_mapping: dict[str, Any] = {}
    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: DiTConfig, hf_config: dict[str, Any], **kwargs) -> None:
        del kwargs
        super().__init__(config=config, hf_config=hf_config)

        arch = config.arch_config
        hidden_size = arch.num_attention_heads * arch.attention_head_dim

        self.out_channels = arch.out_channels
        self.patch_size = arch.patch_size
        self.guidance_embeds = arch.guidance_embeds
        self.num_layers = arch.num_layers
        self.num_single_layers = arch.num_single_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.attention_head_dim = arch.attention_head_dim

        # Input projections
        self.x_embedder = nn.Linear(arch.in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(arch.joint_attention_dim, hidden_size, bias=True)

        # RoPE
        self.pos_embed = EmbedND(
            dim=arch.attention_head_dim,
            theta=10000,
            axes_dim=arch.axes_dim_rope,
        )

        # Timestep/guidance/text conditioning
        if arch.guidance_embeds:
            self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=hidden_size,
                pooled_projection_dim=arch.pooled_projection_dim,
            )
        else:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=hidden_size,
                pooled_projection_dim=arch.pooled_projection_dim,
            )

        # Double-stream blocks
        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(hidden_size, arch.num_attention_heads, arch.attention_head_dim)
            for _ in range(arch.num_layers)
        ])

        # Single-stream blocks
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(hidden_size, arch.num_attention_heads, arch.attention_head_dim)
            for _ in range(arch.num_single_layers)
        ])

        # Output
        self.norm_out = FluxAdaLayerNormContinuous(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, arch.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.__post_init__()

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h = height // 2
        w = width // 2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(-1, 3)

    @staticmethod
    def _prepare_text_ids(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(seq_len, 3, device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H/2 * W/2, C*4)
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(B, (H // 2) * (W // 2), C * 4)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # (B, H/2 * W/2, C*4) -> (B, C, H, W)
        B, _, C4 = latents.shape
        C = C4 // 4
        h, w = height // 2, width // 2
        latents = latents.view(B, h, w, C, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(B, C, height, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> tuple | FluxTransformer2DModelOutput:
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states (T5 sequence) must be provided")
        if pooled_projections is None:
            raise ValueError("pooled_projections (CLIP pooled) must be provided")
        if timestep is None:
            raise ValueError("timestep must be provided")

        if timestep.dim() == 0:
            timestep = timestep[None]
        if timestep.dim() > 1:
            timestep = timestep.reshape(-1)
        if timestep.shape[0] == 1 and hidden_states.shape[0] > 1:
            timestep = timestep.expand(hidden_states.shape[0])

        # hidden_states already packed: (B, seq, 64)
        # If 4D input (B, C, H, W) pack here
        original_shape = None
        if hidden_states.dim() == 4:
            original_shape = hidden_states.shape
            _, _, H, W = original_shape
            # Build positional ids on-the-fly if not provided
            if img_ids is None:
                img_ids = self._prepare_latent_image_ids(H, W, hidden_states.device, hidden_states.dtype)
            if txt_ids is None:
                txt_ids = self._prepare_text_ids(encoder_hidden_states.shape[1], hidden_states.device,
                                                 hidden_states.dtype)
            hidden_states = self._pack_latents(hidden_states)
        else:
            H = W = None

        # Build img_ids / txt_ids from hidden sequence shape if missing
        if img_ids is None:
            img_seq = hidden_states.shape[1]
            side = int(math.isqrt(img_seq))
            img_ids = self._prepare_latent_image_ids(side * 2, side * 2, hidden_states.device, hidden_states.dtype)
        if txt_ids is None:
            txt_ids = self._prepare_text_ids(encoder_hidden_states.shape[1], hidden_states.device,
                                             hidden_states.dtype)

        # Timestep + pooled text conditioning
        if self.guidance_embeds:
            if guidance is None:
                guidance = torch.full((hidden_states.shape[0],), 3.5, device=hidden_states.device,
                                      dtype=hidden_states.dtype)
            if guidance.dim() == 0:
                guidance = guidance[None].expand(hidden_states.shape[0])
            temb = self.time_text_embed(timestep, pooled_projections, guidance)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)

        # Embed inputs
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # RoPE: joint ids for [text, image]
        ids = torch.cat([txt_ids, img_ids], dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # Double-stream blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # Merge text and image for single-stream blocks
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Single-stream blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # Drop text tokens, keep image tokens
        txt_len = encoder_hidden_states.shape[1]
        hidden_states = hidden_states[:, txt_len:]

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpack latents
        if original_shape is not None:
            _, _, H, W = original_shape
            hidden_states = self._unpack_latents(hidden_states, H, W)
        else:
            img_seq = hidden_states.shape[1]
            side = int(math.isqrt(img_seq))
            hidden_states = self._unpack_latents(hidden_states, side * 2, side * 2)

        if not return_dict:
            return (hidden_states,)
        return FluxTransformer2DModelOutput(sample=hidden_states)


EntryClass = FluxTransformer2DModel
