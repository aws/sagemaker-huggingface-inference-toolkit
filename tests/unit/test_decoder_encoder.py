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

import pytest
from transformers.testing_utils import require_torch

from mms.service import PredictionException
from PIL import Image
from sagemaker_huggingface_inference_toolkit import decoder_encoder


ENCODE_JSON_INPUT = {"upper": [1425], "lower": [576], "level": [2], "datetime": ["2012-08-08 15:30"]}
ENCODE_CSV_INPUT = [
    {"answer": "Nuremberg", "end": 42, "score": 0.9926825761795044, "start": 33},
    {"answer": "Berlin is the capital of Germany", "end": 32, "score": 0.26097726821899414, "start": 0},
]

DECODE_JSON_INPUT = {"inputs": "My name is Wolfgang and I live in Berlin"}
DECODE_CSV_INPUT = "question,context\r\nwhere do i live?,My name is Philipp and I live in Nuremberg\r\nwhere is Berlin?,Berlin is the capital of Germany"

CONTENT_TYPE = "application/json"


def test_decode_json():
    decoded_data = decoder_encoder.decode_json(json.dumps(DECODE_JSON_INPUT))
    assert decoded_data == DECODE_JSON_INPUT


def test_decode_csv():
    decoded_data = decoder_encoder.decode_csv(DECODE_CSV_INPUT)
    assert decoded_data == {
        "inputs": [
            {"question": "where do i live?", "context": "My name is Philipp and I live in Nuremberg"},
            {"question": "where is Berlin?", "context": "Berlin is the capital of Germany"},
        ]
    }
    text_classification_input = "inputs\r\nI love you\r\nI like you"
    decoded_data = decoder_encoder.decode_csv(text_classification_input)
    assert decoded_data == {"inputs": ["I love you", "I like you"]}


def test_decode_image():
    image_files_path = os.path.join(os.getcwd(), "tests/resources/image")

    for image_file in os.listdir(image_files_path):
        image_bytes = open(os.path.join(image_files_path, image_file), "rb").read()
        decoded_data = decoder_encoder.decode_image(bytearray(image_bytes))

        assert isinstance(decoded_data, dict)
        assert isinstance(decoded_data["inputs"], Image.Image)


def test_decode_audio():
    audio_files_path = os.path.join(os.getcwd(), "tests/resources/audio")

    for audio_file in os.listdir(audio_files_path):
        audio_bytes = open(os.path.join(audio_files_path, audio_file), "rb").read()
        decoded_data = decoder_encoder.decode_audio(bytearray(audio_bytes))

        assert {"inputs": audio_bytes} == decoded_data


def test_decode_csv_without_header():
    with pytest.raises(PredictionException):
        decoder_encoder.decode_csv(
            "where do i live?,My name is Philipp and I live in Nuremberg\r\nwhere is Berlin?,Berlin is the capital of Germany"
        )


def test_encode_json():
    encoded_data = decoder_encoder.encode_json(ENCODE_JSON_INPUT)
    assert json.loads(encoded_data) == ENCODE_JSON_INPUT


@require_torch
def test_encode_json_torch():
    import torch

    DATA = [1, 0.5, 5.0]
    encoded_data = decoder_encoder.encode_json({"data": torch.tensor(DATA)})
    assert json.loads(encoded_data) == {"data": DATA}


def test_encode_csv():
    decoded_data = decoder_encoder.encode_csv(ENCODE_CSV_INPUT)
    assert (
        decoded_data
        == "answer,end,score,start\r\nNuremberg,42,0.9926825761795044,33\r\nBerlin is the capital of Germany,32,0.26097726821899414,0\r\n"
    )


def test_decode_content_type():
    decoded_content_type = decoder_encoder._decoder_map[CONTENT_TYPE]
    assert decoded_content_type == decoder_encoder.decode_json


def test_encode_content_type():
    encoded_content_type = decoder_encoder._encoder_map[CONTENT_TYPE]
    assert encoded_content_type == decoder_encoder.encode_json


def test_decode():
    decode = decoder_encoder.decode(json.dumps(DECODE_JSON_INPUT), CONTENT_TYPE)
    assert decode == DECODE_JSON_INPUT


def test_encode():
    encode = decoder_encoder.encode(ENCODE_JSON_INPUT, CONTENT_TYPE)
    assert json.loads(encode) == ENCODE_JSON_INPUT
