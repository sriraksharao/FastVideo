# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import SD35TimestepPreparationStage
from fastvideo.pipelines.stages.flux_stages import (
    FluxConditioningStage,
    FluxDecodingStage,
    FluxDenoisingStage,
    FluxLatentPreparationStage,
)

logger = init_logger(__name__)


class FluxPipeline(ComposedPipelineBase):
    """FLUX.1-dev and FLUX.1-schnell text-to-image pipeline.

    Treated as single-frame generation (num_frames=1) inside FastVideo's
    5-D latent convention (B, C, 1, H, W).
    """

    _required_config_modules = [
        "scheduler",
        "transformer",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        te_cfgs = list(fastvideo_args.pipeline_config.text_encoder_configs)
        if len(te_cfgs) >= 1:
            # CLIP
            te_cfgs[0].tokenizer_kwargs.setdefault("padding", "max_length")
            te_cfgs[0].tokenizer_kwargs.setdefault("max_length", 77)
            te_cfgs[0].tokenizer_kwargs.setdefault("truncation", True)
            te_cfgs[0].tokenizer_kwargs.setdefault("return_tensors", "pt")
        if len(te_cfgs) >= 2:
            # T5
            te_cfgs[1].tokenizer_kwargs.setdefault("padding", "max_length")
            te_cfgs[1].tokenizer_kwargs.setdefault("max_length", 512)
            te_cfgs[1].tokenizer_kwargs.setdefault("truncation", True)
            te_cfgs[1].tokenizer_kwargs.setdefault("return_tensors", "pt")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder"), self.get_module("text_encoder_2")],
                tokenizers=[self.get_module("tokenizer"), self.get_module("tokenizer_2")],
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=SD35TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=FluxLatentPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=FluxConditioningStage(
                text_encoders=[self.get_module("text_encoder"), self.get_module("text_encoder_2")],
                tokenizers=[self.get_module("tokenizer"), self.get_module("tokenizer_2")],
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=FluxDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=FluxDecodingStage(vae=self.get_module("vae")),
        )


EntryClass = FluxPipeline
