# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.utils import PRECISION_TO_TYPE


# ---------------------------------------------------------------------------
# Latent preparation
# ---------------------------------------------------------------------------

class FluxLatentPreparationStage(PipelineStage):
    """Initialise noise latents for Flux (4D, uncompressed channels)."""

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.height is None or batch.width is None:
            raise ValueError("height and width must be set before FluxLatentPreparationStage")

        if isinstance(batch.prompt, list):
            batch_size = len(batch.prompt)
        elif batch.prompt is not None:
            batch_size = 1
        else:
            if not batch.prompt_embeds:
                raise ValueError("prompt or prompt_embeds must be provided")
            batch_size = batch.prompt_embeds[0].shape[0]

        batch_size *= batch.num_videos_per_prompt

        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        device = get_local_torch_device()

        vae_spatial = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        # Flux VAE: 16 latent channels
        in_channels_vae = 16
        h_lat = batch.height // vae_spatial
        w_lat = batch.width // vae_spatial

        # Ensure divisible by 2 (for 2x2 packing)
        if h_lat % 2 != 0 or w_lat % 2 != 0:
            raise ValueError(f"Latent height/width must be even, got ({h_lat}, {w_lat}). "
                             f"Use height/width divisible by {vae_spatial * 2}.")

        shape = (batch_size, in_channels_vae, 1, h_lat, w_lat)

        latents = batch.latents
        if latents is None:
            latents = randn_tensor(shape, generator=batch.generator, device=device, dtype=dtype)
            if hasattr(self.scheduler, "init_noise_sigma"):
                latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device=device, dtype=dtype)

        batch.latents = latents
        batch.raw_latent_shape = shape
        return batch


# ---------------------------------------------------------------------------
# Conditioning (build encoder_hidden_states and pooled_projections)
# ---------------------------------------------------------------------------

class FluxConditioningStage(PipelineStage):
    """Assemble T5 sequence + CLIP pooled embeddings for Flux."""

    def __init__(self, text_encoders: list, tokenizers: list) -> None:
        super().__init__()
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

    @staticmethod
    def _tokenize_and_encode(
        text_encoder,
        tokenizer,
        prompts: str | list[str],
        tok_kwargs: dict[str, Any],
        device: torch.device,
    ) -> Any:
        texts = [prompts] if isinstance(prompts, str) else prompts
        enc = tokenizer(texts, **tok_kwargs)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if len(batch.prompt_embeds) < 2:
            raise ValueError(
                f"FluxConditioningStage expects 2 prompt_embeds (CLIP pooled + T5 seq), got {len(batch.prompt_embeds)}"
            )

        device = get_local_torch_device()
        dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]

        # prompt_embeds[0] = CLIP pooled (768,)
        # prompt_embeds[1] = T5 sequence (seq_len, 4096)
        clip_pooled = batch.prompt_embeds[0].to(device=device, dtype=dtype)
        t5_seq = batch.prompt_embeds[1].to(device=device, dtype=dtype)

        batch.extra["flux_encoder_hidden_states"] = t5_seq
        batch.extra["flux_pooled_projections"] = clip_pooled

        if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
            neg_clip = batch.negative_prompt_embeds[0].to(device=device, dtype=dtype)
            neg_t5 = batch.negative_prompt_embeds[1].to(device=device, dtype=dtype)
            batch.extra["flux_neg_encoder_hidden_states"] = neg_t5
            batch.extra["flux_neg_pooled_projections"] = neg_clip

        return batch


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

class FluxDenoisingStage(PipelineStage):
    """Flow-matching denoising loop for Flux (image only, 5D latents)."""

    def __init__(self, transformer, scheduler) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.timesteps is None:
            raise ValueError("timesteps must be set before FluxDenoisingStage")
        if batch.latents is None:
            raise ValueError("latents must be set before FluxDenoisingStage")

        encoder_hidden_states: torch.Tensor = batch.extra["flux_encoder_hidden_states"]
        pooled_projections: torch.Tensor = batch.extra["flux_pooled_projections"]

        timesteps = batch.timesteps
        latents = batch.latents  # (B, C, 1, H, W) from latent prep
        guidance_scale = float(fastvideo_args.pipeline_config.guidance_scale
                               if hasattr(fastvideo_args.pipeline_config, "guidance_scale") else 3.5)

        cfg = fastvideo_args.pipeline_config
        has_guidance = getattr(cfg.dit_config.arch_config, "guidance_embeds", False)

        dtype = PRECISION_TO_TYPE[cfg.dit_precision]
        autocast_enabled = (dtype != torch.float32) and not fastvideo_args.disable_autocast
        device = get_local_torch_device()

        # Build static positional ids from latent shape
        _, _, _, h_lat, w_lat = latents.shape
        img_ids = self.transformer._prepare_latent_image_ids(h_lat, w_lat, device, dtype)
        txt_ids = self.transformer._prepare_text_ids(encoder_hidden_states.shape[1], device, dtype)

        generator = batch.generator[0] if isinstance(batch.generator, list) else batch.generator

        for t in timesteps:
            # Squeeze temporal dim: (B, C, H, W)
            latents_4d = latents.squeeze(2)

            if batch.do_classifier_free_guidance and "flux_neg_encoder_hidden_states" in batch.extra:
                neg_enc = batch.extra["flux_neg_encoder_hidden_states"]
                neg_pool = batch.extra["flux_neg_pooled_projections"]
                latent_input = torch.cat([latents_4d] * 2)
                enc_input = torch.cat([neg_enc, encoder_hidden_states])
                pool_input = torch.cat([neg_pool, pooled_projections])
            else:
                latent_input = latents_4d
                enc_input = encoder_hidden_states
                pool_input = pooled_projections

            latent_input = self.scheduler.scale_model_input(latent_input, t)
            timestep = t.expand(latent_input.shape[0])

            # Pack latents for Flux
            packed = self.transformer._pack_latents(latent_input)

            guidance_t = (
                torch.full((latent_input.shape[0],), guidance_scale, device=device, dtype=dtype)
                if has_guidance else None
            )

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=autocast_enabled and device.type == "cuda",
            ):
                noise_pred = self.transformer(
                    hidden_states=packed,
                    encoder_hidden_states=enc_input,
                    pooled_projections=pool_input,
                    timestep=timestep / 1000.0,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance_t,
                    return_dict=False,
                )[0]

            # Unpack
            noise_pred = self.transformer._unpack_latents(noise_pred, h_lat, w_lat)

            if batch.do_classifier_free_guidance and "flux_neg_encoder_hidden_states" in batch.extra:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            noise_pred_5d = noise_pred.unsqueeze(2)
            latents = self.scheduler.step(noise_pred_5d, t, latents,
                                          return_dict=False,
                                          generator=generator)[0]

        batch.latents = latents
        return batch


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class FluxDecodingStage(PipelineStage):
    """Decode Flux latents using AutoencoderKL."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    @staticmethod
    def _denormalize(latents: torch.Tensor, vae) -> torch.Tensor:
        cfg = getattr(vae, "config", None)
        sf = getattr(cfg, "scaling_factor", None) if cfg is not None else None
        sh = getattr(cfg, "shift_factor", None) if cfg is not None else None

        if sf is None and hasattr(vae, "scaling_factor"):
            sf = vae.scaling_factor
        if sh is None and hasattr(vae, "shift_factor"):
            sh = vae.shift_factor

        if sf is not None:
            latents = latents / (sf.to(latents.device, latents.dtype) if isinstance(sf, torch.Tensor) else sf)
        if sh is not None:
            latents = latents + (sh.to(latents.device, latents.dtype) if isinstance(sh, torch.Tensor) else sh)
        return latents

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if batch.latents is None:
            raise ValueError("latents must be set before FluxDecodingStage")

        device = get_local_torch_device()
        latents = batch.latents.to(device).squeeze(2)  # (B, C, H, W)

        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        autocast_enabled = (vae_dtype != torch.float32) and not fastvideo_args.disable_autocast

        latents = self._denormalize(latents, self.vae)

        with torch.autocast(
            device_type="cuda",
            dtype=vae_dtype,
            enabled=autocast_enabled and device.type == "cuda",
        ):
            if not autocast_enabled:
                latents = latents.to(dtype=vae_dtype)
            dec = self.vae.decode(latents)
            image = dec.sample if hasattr(dec, "sample") else dec[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        batch.output = image.unsqueeze(2).detach().float().cpu()
        return batch
