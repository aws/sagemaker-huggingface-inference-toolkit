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
import base64
import csv
import datetime
import json
from io import BytesIO, StringIO

import numpy as np
from sagemaker_inference import errors
from sagemaker_inference.decoder import _npy_to_numpy
from sagemaker_inference.encoder import _array_to_npy

from mms.service import PredictionException
from PIL import Image
from sagemaker_huggingface_inference_toolkit import content_types


def decode_json(content):
    return json.loads(content)


def decode_csv(string_like):  # type: (str) -> np.array
    """Convert a CSV object to a dictonary with list attributes.

    Args:
        string_like (str): CSV string.
    Returns:
        (dict): dictonatry for input
    """
    stream = StringIO(string_like)
    # detects if the incoming csv has headers
    if not any(header in string_like.splitlines()[0].lower() for header in ["question", "context", "inputs"]):
        raise PredictionException(
            "You need to provide the correct CSV with Header columns to use it with the inference toolkit default handler.",
            400,
        )
    # reads csv as io
    request_list = list(csv.DictReader(stream))
    if "inputs" in request_list[0].keys():
        return {"inputs": [entry["inputs"] for entry in request_list]}
    else:
        return {"inputs": request_list}


def decode_image(bpayload: bytearray):
    """Convert a .jpeg / .png / .tiff... object to a proper inputs dict.
    Args:
        bpayload (bytes): byte stream.
    Returns:
        (dict): dictonatry for input
    """
    image = Image.open(BytesIO(bpayload)).convert("RGB")
    return {"inputs": image}


def decode_audio(bpayload: bytearray):
    """Convert a .wav / .flac / .mp3 object to a proper inputs dict.
    Args:
        bpayload (bytes): byte stream.
    Returns:
        (dict): dictonatry for input
    """

    return {"inputs": bytes(bpayload)}


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
        elif isinstance(obj, Image.Image):
            with BytesIO() as out:
                obj.save(out, format="PNG")
                png_string = out.getvalue()
                return base64.b64encode(png_string).decode("utf-8")
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


def encode_csv(content):  # type: (str) -> np.array
    """Convert the result of a transformers pipeline to CSV.
    Args:
        content (dict | list): result of transformers pipeline.
    Returns:
        (str): object serialized to CSV
    """
    stream = StringIO()
    if not isinstance(content, list):
        content = list(content)

    column_header = content[0].keys()
    writer = csv.DictWriter(stream, column_header)

    writer.writeheader()
    writer.writerows(content)
    return stream.getvalue()


_encoder_map = {
    content_types.NPY: _array_to_npy,
    content_types.CSV: encode_csv,
    content_types.JSON: encode_json,
}
_decoder_map = {
    content_types.NPY: _npy_to_numpy,
    content_types.CSV: decode_csv,
    content_types.JSON: decode_json,
    # image mime-types
    content_types.JPEG: decode_image,
    content_types.PNG: decode_image,
    content_types.TIFF: decode_image,
    content_types.BMP: decode_image,
    content_types.GIF: decode_image,
    content_types.WEBP: decode_image,
    content_types.X_IMAGE: decode_image,
    # audio mime-types
    content_types.FLAC: decode_audio,
    content_types.MP3: decode_audio,
    content_types.WAV: decode_audio,
    content_types.OGG: decode_audio,
    content_types.X_AUDIO: decode_audio,
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
    except PredictionException as pred_err:
        raise pred_err


def encode(content, content_type=content_types.JSON):
    """
    Encode an 🤗 Transformers object in a specific content_type.
    """
    try:
        encoder = _encoder_map[content_type]
        return encoder(content)
    except KeyError:
        raise errors.UnsupportedFormatError(content_type)
