# Copyright 2021 The HuggingFace Team, Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
import tempfile

import pytest
from transformers.testing_utils import require_torch

from sagemaker_huggingface_inference_toolkit.optimum_utils import (
    get_input_shapes,
    get_optimum_neuron_pipeline,
    is_optimum_neuron_available,
)
from sagemaker_huggingface_inference_toolkit.transformers_utils import _load_model_from_hub


require_inferentia = pytest.mark.skipif(
    not is_optimum_neuron_available(),
    reason="Skipping tests, since optimum neuron is not available or not running on inf2 instances.",
)


REMOTE_NOT_CONVERTED_MODEL = "hf-internal-testing/tiny-random-BertModel"
REMOTE_CONVERTED_MODEL = "optimum/tiny_random_bert_neuron"
TASK = "text-classification"


@require_torch
@require_inferentia
def test_not_supported_task():
    os.environ["HF_TASK"] = "not-supported-task"
    with pytest.raises(Exception) as e:
        get_optimum_neuron_pipeline(task=TASK, model_dir=os.getcwd())


@require_torch
@require_inferentia
def test_get_input_shapes_from_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=REMOTE_CONVERTED_MODEL,
            model_dir=tmpdirname,
        )
        input_shapes = get_input_shapes(model_dir=storage_folder)
        assert input_shapes["batch_size"] == 1
        assert input_shapes["sequence_length"] == 16


@require_torch
@require_inferentia
def test_get_input_shapes_from_env():
    os.environ["OPTIMUM_NEURON_BATCH_SIZE"] = "4"
    os.environ["OPTIMUM_NEURON_SEQUENCE_LENGTH"] = "32"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=REMOTE_NOT_CONVERTED_MODEL,
            model_dir=tmpdirname,
        )
        input_shapes = get_input_shapes(model_dir=storage_folder)
        assert input_shapes["batch_size"] == 4
        assert input_shapes["sequence_length"] == 32


@require_torch
@require_inferentia
def test_get_optimum_neuron_pipeline_from_converted_model():
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.system(
            f"optimum-cli export neuron --model philschmid/tiny-distilbert-classification --sequence_length 32 --batch_size 1 {tmpdirname}"
        )
        pipe = get_optimum_neuron_pipeline(task=TASK, model_dir=tmpdirname)
        r = pipe("This is a test")

        assert r[0]["score"] > 0.0
        assert isinstance(r[0]["label"], str)


@require_torch
@require_inferentia
def test_get_optimum_neuron_pipeline_from_non_converted_model():
    os.environ["OPTIMUM_NEURON_SEQUENCE_LENGTH"] = "32"
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=REMOTE_NOT_CONVERTED_MODEL,
            model_dir=tmpdirname,
        )
        pipe = get_optimum_neuron_pipeline(task=TASK, model_dir=storage_folder)
        r = pipe("This is a test")

        assert r[0]["score"] > 0.0
        assert isinstance(r[0]["label"], str)
