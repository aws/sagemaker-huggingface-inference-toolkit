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
import tempfile

from transformers.testing_utils import require_torch, slow

from PIL import Image
from sagemaker_huggingface_inference_toolkit.diffusers_utils import SMDiffusionPipelineForText2Image
from sagemaker_huggingface_inference_toolkit.transformers_utils import _load_model_from_hub, get_pipeline


@require_torch
def test_get_diffusers_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_model_from_hub(
            "hf-internal-testing/tiny-stable-diffusion-torch",
            tmpdirname,
        )
        pipe = get_pipeline("text-to-image", -1, storage_dir)
        assert isinstance(pipe, SMDiffusionPipelineForText2Image)


@slow
@require_torch
def test_pipe_on_gpu():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_model_from_hub(
            "hf-internal-testing/tiny-stable-diffusion-torch",
            tmpdirname,
        )
        pipe = get_pipeline("text-to-image", 0, storage_dir)
        assert pipe.device.type == "cuda"


@require_torch
def test_text_to_image_task():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_model_from_hub("hf-internal-testing/tiny-stable-diffusion-torch", tmpdirname)
        pipe = get_pipeline("text-to-image", -1, storage_dir)
        res = pipe("Lets create an embedding")
        assert isinstance(res, Image.Image)
