# SPDX-License-Identifier: Apache-2.0

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
)

FLUX_DEV = InferencePreset(
    name="flux_dev",
    version=1,
    model_family="flux",
    description="FLUX.1-dev text-to-image (28 steps, guidance=3.5)",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE,),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "negative_prompt": "",
    },
)

FLUX_SCHNELL = InferencePreset(
    name="flux_schnell",
    version=1,
    model_family="flux_schnell",
    description="FLUX.1-schnell text-to-image (4 steps, distilled)",
    workload_type="t2i",
    stage_schemas=(_DENOISE_STAGE,),
    defaults={
        "height": 1024,
        "width": 1024,
        "num_frames": 1,
        "fps": 1,
        "seed": 0,
        "guidance_scale": 0.0,
        "num_inference_steps": 4,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (FLUX_DEV, FLUX_SCHNELL)
