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

# Build release wheels as follows
# $ SM_HF_TOOLKIT_RELEASE=1 python setup.py bdist_wheel build
# $ twine upload --repository-url https://test.pypi.org/legacy/ dist/* # upload to test.pypi
# Test the wheel by downloading from test.pypi
# $ pip install -i https://test.pypi.org/simple/ sagemaker-huggingface-inference-toolkit==<version>
# Once test is complete
# Upload the wheel to pypi
# $ twine upload dist/*


from __future__ import absolute_import
import os
from datetime import date
from setuptools import find_packages, setup

# We don't declare our dependency on transformers here because we build with
# different packages for different variants

VERSION = "2.0.0"


# Ubuntu packages
# libsndfile1-dev: torchaudio requires the development version of the libsndfile package which can be installed via a system package manager. On Ubuntu it can be installed as follows: apt install libsndfile1-dev
# ffmpeg: ffmpeg is required for audio processing. On Ubuntu it can be installed as follows: apt install ffmpeg
# libavcodec-extra : libavcodec-extra  inculdes additional codecs for ffmpeg

install_requires = [
    "sagemaker-inference>=1.5.11",
    "huggingface_hub>=0.0.8",
    "retrying",
    "numpy",
    # vision
    "Pillow",
    # speech + torchaudio
    "librosa",
    "pyctcdecode>=0.3.0",
    "phonemizer",
]

extras = {}

# Hugging Face specific dependencies
extras["transformers"] = ["transformers[sklearn,sentencepiece]>=4.17.0"]

# framework specific dependencies
extras["torch"] = ["torch>=1.8.0", "torchaudio"]
extras["tensorflow"] = ["tensorflow>=2.4.0"]

# MMS Server dependencies
extras["mms"] = ["multi-model-server>=1.1.4", "retrying"]


extras["test"] = [
    "pytest",
    "pytest-xdist",
    "parameterized",
    "psutil",
    "datasets",
    "pytest-sugar",
    "black==21.4b0",
    "sagemaker",
    "boto3",
    "mock==2.0.0",
]

extras["benchmark"] = ["boto3", "locust"]

extras["quality"] = [
    "black==21.4b0",
    "isort>=5.5.4",
    "flake8>=3.8.3",
]

extras["dev"] = extras["transformers"] + extras["mms"] + extras["torch"] + extras["tensorflow"]

setup(
    name="sagemaker-huggingface-inference-toolkit",
    version=VERSION,
    # if os.getenv("SM_HF_TOOLKIT_RELEASE") is not None
    # else VERSION + "b" + str(date.today()).replace("-", ""),
    author="HuggingFace and Amazon Web Services",
    description="Open source library for running inference workload with Hugging Face Deep Learning Containers on "
    "Amazon SageMaker.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep-learning transformer pytorch tensorflow BERT GPT GPT-2 AWS Amazon SageMaker Cloud",
    url="https://github.com/aws/sagemaker-huggingface-inference-toolkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require=extras,
    entry_points={"console_scripts": "serve=sagemaker_huggingface_inference_toolkit.serving:main"},
    python_requires=">=3.6.0",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
