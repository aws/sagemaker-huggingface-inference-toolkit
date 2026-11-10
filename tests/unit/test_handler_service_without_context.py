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
import json
import os
import tempfile
from unittest.mock import Mock

import pytest
from sagemaker_inference import content_types
from transformers.testing_utils import require_torch, slow

from mms.context import Context, RequestProcessor
from mms.metrics.metrics_store import MetricsStore
from mms.service import PredictionException
from sagemaker_huggingface_inference_toolkit import handler_service
from sagemaker_huggingface_inference_toolkit.transformers_utils import _load_model_from_hub, get_pipeline

TASK = "text-classification"
MODEL = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
INPUT = {"inputs": "My name is Wolfgang and I live in Berlin"}
OUTPUT = [
    {"word": "Wolfgang", "score": 0.99, "entity": "I-PER", "index": 4, "start": 11, "end": 19},
    {"word": "Berlin", "score": 0.99, "entity": "I-LOC", "index": 9, "start": 34, "end": 40},
]


@pytest.fixture()
def inference_handler():
    return handler_service.HuggingFaceHandlerService()


def test_get_device_cpu(inference_handler):
    device = inference_handler.get_device()
    assert device == -1


@slow
def test_get_device_gpu(inference_handler):
    device = inference_handler.get_device()
    assert device > -1


@require_torch
def test_test_initialize(inference_handler):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )
        CONTEXT = Context(MODEL, storage_folder, {}, 1, -1, "1.1.4")

        inference_handler.initialize(CONTEXT)
        assert inference_handler.initialized is True


@require_torch
def test_handle(inference_handler):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )
        CONTEXT = Context(MODEL, storage_folder, {}, 1, -1, "1.1.4")
        CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
        CONTEXT.metrics = MetricsStore(1, MODEL)

        inference_handler.initialize(CONTEXT)
        json_data = json.dumps(INPUT)
        prediction = inference_handler.handle([{"body": json_data.encode()}], CONTEXT)
        assert "output" in prediction[0]


@require_torch
def test_load(inference_handler):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )
        # test with automatic infer
        if "HF_TASK" in os.environ:
            del os.environ["HF_TASK"]
        hf_pipeline_without_task = inference_handler.load(storage_folder)
        assert hf_pipeline_without_task.task == "token-classification"

        # test with automatic infer
        os.environ["HF_TASK"] = "text-classification"
        hf_pipeline_with_task = inference_handler.load(storage_folder)
        assert hf_pipeline_with_task.task == "text-classification"


def test_preprocess(inference_handler):
    json_data = json.dumps(INPUT)
    decoded_input_data = inference_handler.preprocess(json_data, content_types.JSON)
    assert "inputs" in decoded_input_data


def test_preprocess_bad_content_type(inference_handler):
    with pytest.raises(json.decoder.JSONDecodeError):
        inference_handler.preprocess(b"", content_types.JSON)


@require_torch
def test_predict(inference_handler):
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_folder = _load_model_from_hub(
            model_id=MODEL,
            model_dir=tmpdirname,
        )
        inference_handler.model = get_pipeline(task=TASK, device=-1, model_dir=storage_folder)
        prediction = inference_handler.predict(INPUT, inference_handler.model)
        assert "label" in prediction[0]
        assert "score" in prediction[0]


def test_postprocess(inference_handler):
    output = inference_handler.postprocess(OUTPUT, content_types.JSON)
    assert isinstance(output, str)


@pytest.mark.parametrize("key", ["custom_generate", "trust_remote_code"])
@pytest.mark.parametrize("value", [True, False, None, "example/model"])
@pytest.mark.parametrize("location", ["top", "generate_kwargs", "nested_dict", "nested_list", "nested_tuple"])
@pytest.mark.parametrize("context", [None, object()])
def test_predict_rejects_unsupported_parameters(inference_handler, key, value, location, context):
    parameters = {key: value}
    if location == "generate_kwargs":
        parameters = {"generate_kwargs": parameters}
    elif location == "nested_dict":
        parameters = {"options": {"nested": parameters}}
    elif location == "nested_list":
        parameters = {"options": ["allowed", [{"nested": parameters}]]}
    elif location == "nested_tuple":
        parameters = {"options": ("allowed", parameters)}
    model = Mock()

    with pytest.raises(PredictionException, match=key) as error:
        inference_handler.predict({"inputs": "Hello", "parameters": parameters}, model, context)

    assert error.value.error_code == 400
    model.assert_not_called()


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {},
        {"max_new_tokens": 4},
        {"generate_kwargs": {"do_sample": False}},
        {"options": ["custom_generate", {"names": ["trust_remote_code"]}]},
    ],
)
def test_predict_passes_supported_parameters(inference_handler, parameters):
    model = Mock(return_value="prediction")
    data = {"inputs": "Hello", "parameters": parameters}

    assert inference_handler.predict(data, model) == "prediction"

    model.assert_called_once_with("Hello", **(parameters or {}))


def test_predict_without_parameters(inference_handler):
    model = Mock(return_value="prediction")

    assert inference_handler.predict({"inputs": "Hello"}, model) == "prediction"

    model.assert_called_once_with("Hello")


@pytest.mark.parametrize("key", ["custom_generate", "trust_remote_code"])
def test_handle_rejects_unsupported_parameters(inference_handler, key):
    inference_handler.initialized = True
    inference_handler.model = Mock()
    context = Context("", "", {}, 1, -1, "1.1.4")
    context.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
    payload = {"inputs": "Hello", "parameters": {"generate_kwargs": {key: True}}}

    with pytest.raises(PredictionException, match=key) as error:
        inference_handler.handle([{"body": json.dumps(payload).encode()}], context)

    assert error.value.error_code == 400
    inference_handler.model.assert_not_called()


def test_validate_and_initialize_user_module(inference_handler):
    model_dir = os.path.join(os.getcwd(), "tests/resources/model_input_predict_output_fn_without_context")
    CONTEXT = Context("", model_dir, {}, 1, -1, "1.1.4")

    inference_handler.initialize(CONTEXT)
    CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
    CONTEXT.metrics = MetricsStore(1, MODEL)

    prediction = inference_handler.handle([{"body": b""}], CONTEXT)
    assert "output" in prediction[0]

    assert inference_handler.load({}) == "Loading inference_tranform_fn.py"


def test_validate_and_initialize_user_module_transform_fn():
    os.environ["SAGEMAKER_PROGRAM"] = "inference_tranform_fn.py"
    inference_handler = handler_service.HuggingFaceHandlerService()
    model_dir = os.path.join(os.getcwd(), "tests/resources/model_transform_fn_without_context")
    CONTEXT = Context("dummy", model_dir, {}, 1, -1, "1.1.4")

    inference_handler.initialize(CONTEXT)
    CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
    CONTEXT.metrics = MetricsStore(1, MODEL)
    assert "output" in inference_handler.handle([{"body": b"dummy"}], CONTEXT)[0]
    assert inference_handler.load({}) == "Loading inference_tranform_fn.py"
    assert inference_handler.transform_fn("model", "dummy", "application/json", "application/json") == "output dummy"


def test_validate_and_initialize_user_module_transform_fn_race_condition():
    os.environ["SAGEMAKER_PROGRAM"] = "inference_tranform_fn.py"
    inference_handler = handler_service.HuggingFaceHandlerService()
    model_dir = os.path.join(os.getcwd(), "tests/resources/model_transform_fn_without_context")
    CONTEXT = Context("dummy", model_dir, {}, 1, -1, "1.1.4")

    # Similuate 2 threads bypassing check in handle() - calling initialize twice
    inference_handler.initialize(CONTEXT)
    inference_handler.initialize(CONTEXT)

    CONTEXT.request_processor = [RequestProcessor({"Content-Type": "application/json"})]
    CONTEXT.metrics = MetricsStore(1, MODEL)
    assert "output" in inference_handler.handle([{"body": b"dummy"}], CONTEXT)[0]
    assert inference_handler.load({}) == "Loading inference_tranform_fn.py"
    assert inference_handler.transform_fn("model", "dummy", "application/json", "application/json") == "output dummy"
