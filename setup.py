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
from __future__ import absolute_import

from setuptools import find_packages, setup

# We don't declare our dependency on transformers here because we build with
# different packages for different variants
install_requires = [
    "sagemaker-inference>=1.5.5",
    "huggingface_hub>=0.0.8",
    "retrying",
    "numpy",
]

extras = {}

# Hugging Face specific dependencies
extras["transformers"] = ["transformers[sklearn,sentencepiece]>=4.5.1"]

# framework specific dependencies
extras["torch"] = ["torch>=1.4.0"]
extras["tensorflow-cpu"] = ["tensorflow-cpu>=2.3"]
extras["tensorflow"] = ["tensorflow>=2.3"]

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

extras["all"] = extras["test"] + extras["quality"] + extras["benchmark"] + extras["transformers"] + extras["mms"]

setup(
    name="sagemaker-huggingface-inference-toolkit",
    version="0.1.0",
    author="HuggingFace and Amazon Web Services",
    description="Open source library for running inference workload with Hugging Face Deep Learning Containers on Amazon SageMaker.",
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
