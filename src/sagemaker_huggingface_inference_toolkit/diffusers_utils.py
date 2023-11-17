# Copyright 2023 The HuggingFace Team, Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import importlib.util
import logging

from transformers.utils.import_utils import is_torch_bf16_gpu_available


logger = logging.getLogger(__name__)

_diffusers = importlib.util.find_spec("diffusers") is not None


def is_diffusers_available():
    return _diffusers


if is_diffusers_available():
    import torch

    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, StableDiffusionPipeline


class SMAutoPipelineForText2Image:
    def __init__(self, model_dir: str, device: str = None):  # needs "cuda" for GPU
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
        device_map = "auto" if device == "cuda" else None

        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=dtype, device_map=device_map)
        # try to use DPMSolverMultistepScheduler
        if isinstance(self.pipeline, StableDiffusionPipeline):
            try:
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            except Exception:
                pass
        self.pipeline.to(device)

    def __call__(
        self,
        prompt,
        **kwargs,
    ):
        # TODO: add support for more images (Reason is correct output)
        if "num_images_per_prompt" in kwargs:
            kwargs.pop("num_images_per_prompt")
            logger.warning("Sending num_images_per_prompt > 1 to pipeline is not supported. Using default value 1.")

        # Call pipeline with parameters
        out = self.pipeline(prompt, num_images_per_prompt=1, **kwargs)
        return out.images[0]


DIFFUSERS_TASKS = {
    "text-to-image": SMAutoPipelineForText2Image,
}


def get_diffusers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    """Get a pipeline for Diffusers models."""
    device = "cuda" if device == 0 else "cpu"
    pipeline = DIFFUSERS_TASKS[task](model_dir=model_dir, device=device)
    return pipeline
