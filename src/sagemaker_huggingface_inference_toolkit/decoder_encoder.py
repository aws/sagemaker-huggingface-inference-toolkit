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
import datetime
import json

import numpy as np
from sagemaker_inference.decoder import (
    _npy_to_numpy,
    _csv_to_numpy,
    _npz_to_sparse,
)
from sagemaker_inference.encoder import (
    _array_to_npy,
    _array_to_csv,
)
from sagemaker_inference import (
    content_types,
    errors,
)


def decode_json(content):
    return json.loads(content)


# https://github.com/automl/SMAC3/issues/453
class _JSONEncoder(json.JSONEncoder):
    """
    custom `JSONEncoder` to make sure float and int64 ar converted
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        else:
            return super(_JSONEncoder, self).default(obj)


def encode_json(content):
    """
    encodes json with custom `JSONEncoder`
    """
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        cls=_JSONEncoder,
        separators=(",", ":"),
    )


_encoder_map = {
    content_types.NPY: _array_to_npy,
    content_types.CSV: _array_to_csv,
    content_types.JSON: encode_json,
}
_decoder_map = {
    content_types.NPY: _npy_to_numpy,
    content_types.CSV: _csv_to_numpy,
    content_types.NPZ: _npz_to_sparse,
    content_types.JSON: decode_json,
}


def decode(content, content_type=content_types.JSON):
    """
    Decodes a specific content_type into an 🤗 Transformers object.
    """
    try:
        decoder = _decoder_map[content_type]
        return decoder(content)
    except KeyError:
        raise errors.UnsupportedFormatError(content_type)


def encode(content, content_type=content_types.JSON):
    """
    Encode an 🤗 Transformers object in a specific content_type.
    """
    try:
        encoder = _encoder_map[content_type]
        return encoder(content)
    except KeyError:
        raise errors.UnsupportedFormatError(content_type)
