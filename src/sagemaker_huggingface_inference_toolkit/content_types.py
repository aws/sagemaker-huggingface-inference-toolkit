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
"""This module contains constants that define MIME content types."""
# Default Mime-Types
JSON = "application/json"
CSV = "text/csv"
OCTET_STREAM = "application/octet-stream"
ANY = "*/*"
NPY = "application/x-npy"
UTF8_TYPES = [JSON, CSV]
# Vision Mime-Types
JPEG = "image/jpeg"
PNG = "image/png"
TIFF = "image/tiff"
BMP = "image/bmp"
GIF = "image/gif"
WEBP = "image/webp"
X_IMAGE = "image/x-image"
VISION_TYPES = [JPEG, PNG, TIFF, BMP, GIF, WEBP, X_IMAGE]
# Speech Mime-Types
FLAC = "audio/x-flac"
MP3 = "audio/mpeg"
WAV = "audio/wave"
OGG = "audio/ogg"
X_AUDIO = "audio/x-audio"
AUDIO_TYPES = [FLAC, MP3, WAV, OGG, X_AUDIO]
