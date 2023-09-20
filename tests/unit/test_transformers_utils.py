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

from transformers import pipeline
from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_tf, require_torch, slow

from sagemaker_huggingface_inference_toolkit.transformers_utils import (
    _build_storage_path,
    _get_framework,
    _is_gpu_available,
    _load_model_from_hub,
    get_pipeline,
    infer_task_from_hub,
    infer_task_from_model_architecture,
    wrap_conversation_pipeline,
)


MODEL = "lysandre/tiny-bert-random"
TASK = "text-classification"
TASK_MODEL = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"

MAIN_REVISION = "main"
REVISION = "eb4c77816edd604d0318f8e748a1c606a2888493"


@require_torch
def test_loading_model_from_hub():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )

        # folder contains all config files and pytorch_model.bin
        folder_contents = os.listdir(storage_folder)
        assert "config.json" in folder_contents


@require_torch
def test_loading_model_from_hub_with_revision():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(model_id=MODEL, model_dir=tmpdirname, revision=REVISION)

        # folder contains all config files and pytorch_model.bin
        assert REVISION in storage_folder
        folder_contents = os.listdir(storage_folder)
        assert "config.json" in folder_contents
        assert "tokenizer_config.json" not in folder_contents


@require_torch
def test_loading_model_safetensor_from_hub_with_revision():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id="hf-internal-testing/tiny-random-bert-safetensors", model_dir=tmpdirname
        )

        folder_contents = os.listdir(storage_folder)
        assert "model.safetensors" in folder_contents


def test_gpu_is_not_available():
    device = _is_gpu_available()
    assert device is False


def test_build_storage_path():
    storage_path = _build_storage_path(model_id=MODEL, model_dir="x")
    assert "__" in storage_path

    storage_path = _build_storage_path(model_id="bert-base-uncased", model_dir="x")
    assert "__" not in storage_path

    storage_path = _build_storage_path(model_id=MODEL, model_dir="x", revision=REVISION)
    assert "__" in storage_path and "." in storage_path


@slow
def test_gpu_available():
    device = _is_gpu_available()
    assert device is True


@require_torch
def test_get_framework_pytorch():
    framework = _get_framework()
    assert framework == "pytorch"


@require_tf
def test_get_framework_tensorflow():
    framework = _get_framework()
    if is_torch_available():
        assert framework == "pytorch"
    else:
        assert framework == "tensorflow"


def test_get_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_model_from_hub(MODEL, tmpdirname)
        pipe = get_pipeline(TASK, -1, storage_dir)
        res = pipe("Life is good, Life is bad")
        assert "score" in res[0]


def test_infer_task_from_hub():
    task = infer_task_from_hub(TASK_MODEL)
    assert task == "token-classification"


def test_infer_task_from_model_architecture():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_model_from_hub(TASK_MODEL, tmpdirname)
        task = infer_task_from_model_architecture(f"{storage_dir}/config.json")
        assert task == "token-classification"


@require_torch
def test_wrap_conversation_pipeline():
    init_pipeline = pipeline(
        "conversational",
        model="microsoft/DialoGPT-small",
        tokenizer="microsoft/DialoGPT-small",
        framework="pt",
    )
    conv_pipe = wrap_conversation_pipeline(init_pipeline)
    data = {
        "past_user_inputs": ["Which movie is the best ?"],
        "generated_responses": ["It's Die Hard for sure."],
        "text": "Can you explain why?",
    }
    res = conv_pipe(data)
    assert "conversation" in res
    assert "generated_text" in res


@require_torch
def test_wrapped_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_model_from_hub("microsoft/DialoGPT-small", tmpdirname)
        conv_pipe = get_pipeline("conversational", -1, storage_dir)
        data = {
            "past_user_inputs": ["Which movie is the best ?"],
            "generated_responses": ["It's Die Hard for sure."],
            "text": "Can you explain why?",
        }
        res = conv_pipe(data)
        assert "conversation" in res
        assert "generated_text" in res
