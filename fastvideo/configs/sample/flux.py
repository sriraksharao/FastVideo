# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class FluxSamplingParam(SamplingParam):
    """Sampling parameters for Flux text-to-image generation."""
    num_inference_steps: int = 20

    # Image dimensions
    height: int = 512
    width: int = 512

    # Guidance scale
    guidance_scale: float = 3.5
