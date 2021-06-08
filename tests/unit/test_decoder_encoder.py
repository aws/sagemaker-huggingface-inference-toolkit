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

from sagemaker_huggingface_inference_toolkit import decoder_encoder


ENCODE_INPUT = {"upper": [1425], "lower": [576], "level": [2], "datetime": ["2012-08-08 15:30"]}
DECODE_INPUT = {"inputs": "My name is Wolfgang and I live in Berlin"}

CONTENT_TYPE = "application/json"


def test_decode_json():
    decoded_data = decoder_encoder.decode_json(json.dumps(DECODE_INPUT))
    assert decoded_data == DECODE_INPUT


def test_encode_json():
    encoded_data = decoder_encoder.encode_json(ENCODE_INPUT)
    assert json.loads(encoded_data) == ENCODE_INPUT


def test_decode_content_type():
    decoded_content_type = decoder_encoder._decoder_map[CONTENT_TYPE]
    assert decoded_content_type == decoder_encoder.decode_json


def test_encode_content_type():
    encoded_content_type = decoder_encoder._encoder_map[CONTENT_TYPE]
    assert encoded_content_type == decoder_encoder.encode_json


def test_decode():
    decode = decoder_encoder.decode(json.dumps(DECODE_INPUT), CONTENT_TYPE)
    assert decode == DECODE_INPUT


def test_encode():
    encode = decoder_encoder.encode(ENCODE_INPUT, CONTENT_TYPE)
    assert json.loads(encode) == ENCODE_INPUT
