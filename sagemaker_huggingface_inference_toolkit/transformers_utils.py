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
import logging
import os
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub.file_download import cached_download, hf_hub_url
from transformers import pipeline
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.pipelines import Pipeline


if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch

from pathlib import Path


logger = logging.getLogger(__name__)

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
FRAMEWORK_MAPPING = {"pytorch": PYTORCH_WEIGHTS_NAME, "tensorflow": TF2_WEIGHTS_NAME}

FILE_LIST_NAMES = [
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "dict.txt",
    "preprocessor_config.json",
    "added_tokens.json",
    "README.md",
    "spiece.model",
    "sentencepiece.bpe.model",
    "sentencepiece.bpe.vocab",
    "sentence.bpe.model",
    "bpe.codes",
    "source.spm",
    "target.spm",
    "spm.model",
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "added_tokens.json",
    "model_args.json",
    "entity_vocab.json",
    "pooling_config.json",
]

REPO_ID_SEPARATOR = "__"

ARCHITECTURES_2_TASK = {
    "TapasForQuestionAnswering": "table-question-answering",
    "ForQuestionAnswering": "question-answering",
    "ForTokenClassification": "token-classification",
    "ForSequenceClassification": "text-classification",
    "ForMultipleChoice": "multiple-choice",
    "ForMaskedLM": "fill-mask",
    "ForCausalLM": "text-generation",
    "ForConditionalGeneration": "text2text-generation",
    "MTModel": "text2text-generation",
    "EncoderDecoderModel": "text2text-generation",
    # Model specific task for backward comp
    "GPT2LMHeadModel": "text-generation",
    "T5WithLMHeadModel": "text2text-generation",
}


HF_API_TOKEN = os.environ.get("HF_API_TOKEN", None)
HF_MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", None)


def _is_gpu_available():
    """
    checks if a gpu is available.
    """
    if is_tf_available():
        return True if len(tf.config.list_physical_devices("GPU")) > 0 else False
    elif is_torch_available():
        return torch.cuda.is_available()
    else:
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )


def _get_framework():
    """
    extracts which DL framework is used for inference, if both are installed use pytorch
    """
    if is_torch_available():
        return "pytorch"
    elif is_tf_available():
        return "tensorflow"
    else:
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )


def _build_storage_path(model_id: str, model_dir: Path, revision: Optional[str] = None):
    """
    creates storage path for hub model based on model_id and revision
    """
    if "/" and revision is None:
        storage_path = os.path.join(model_dir, model_id.replace("/", REPO_ID_SEPARATOR))
    elif "/" and revision is not None:
        storage_path = os.path.join(model_dir, model_id.replace("/", REPO_ID_SEPARATOR) + "." + revision)
    elif revision is not None:
        storage_path = os.path.join(model_dir, model_id + "." + revision)
    else:
        storage_path = os.path.join(model_dir, model_id)
    return storage_path


def _load_model_from_hub(
    model_id: str, model_dir: Path, revision: Optional[str] = None, use_auth_token: Optional[str] = None
):
    """
    Downloads a model repository at the specified revision from the Hugging Face Hub.
    All files are nested inside a folder in order to keep their actual filename
    relative to that folder. `org__model.revision`
    """
    logger.warn(
        "This is an experimental beta features, which allows downloading model from the Hugging Face Hub on start up. "
        "It loads the model defined in the env var `HF_MODEL_ID`"
    )
    # get all files from repository
    _api = HfApi()
    model_info = _api.model_info(repo_id=model_id, revision=revision, token=use_auth_token)
    os.makedirs(model_dir, exist_ok=True)

    # extracts base framework
    framework = _get_framework()

    # creates directory for saved model based on revision and model
    storage_folder = _build_storage_path(model_id, model_dir, revision)
    os.makedirs(storage_folder, exist_ok=True)

    # filters files to download
    download_file_list = [
        file.rfilename
        for file in model_info.siblings
        if file.rfilename in FILE_LIST_NAMES + [FRAMEWORK_MAPPING[framework]]
    ]

    # download files to storage_folder and removes cache
    for file in download_file_list:
        url = hf_hub_url(model_id, filename=file, revision=revision)

        path = cached_download(url, cache_dir=storage_folder, force_filename=file, use_auth_token=use_auth_token)

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder


def infer_task_from_model_architecture(model_config_path: str, architecture_index=0) -> str:
    """
    Infer task from `config.json` of trained model. It is not guaranteed to the detect, e.g. some models implement multiple architectures or
    trainend on different tasks https://huggingface.co/facebook/bart-large/blob/main/config.json. Should work for every on Amazon SageMaker fine-tuned model.
    It is always recommended to set the task through the env var `TASK`.
    """
    with open(model_config_path, "r+") as config_file:
        config = json.loads(config_file.read())
        architecture = config.get("architectures", [None])[architecture_index]

    task = None
    for arch_options in ARCHITECTURES_2_TASK:
        if architecture.endswith(arch_options):
            task = ARCHITECTURES_2_TASK[arch_options]

    if task is None:
        raise ValueError(
            f"Task couldn't be inferenced from {architecture}."
            f"Inference Toolkit can only inference tasks from architectures ending with {list(ARCHITECTURES_2_TASK.keys())}."
            "Use env `HF_TASK` to define your task."
        )
    return task


def infer_task_from_hub(model_id: str, revision: Optional[str] = None, use_auth_token: Optional[str] = None) -> str:
    """
    Infer task from Hub by extracting `pipeline_tag` for model_info.
    """
    _api = HfApi()
    model_info = _api.model_info(repo_id=model_id, revision=revision, token=use_auth_token)
    if model_info.pipeline_tag is not None:
        return model_info.pipeline_tag
    else:
        raise ValueError(
            f"Task couldn't be inferenced from {model_info.pipeline_tag}." "Use env `HF_TASK` to define your task."
        )


def get_pipeline(task: str, device: int, model_dir: Path, **kwargs) -> Pipeline:
    """
    create pipeline class for a specific task based on local saved model
    """
    if task is None:
        raise EnvironmentError(
            "The task for this model is not set: Please set one: https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined"
        )

    hf_pipeline = pipeline(task=task, model=model_dir, tokenizer=model_dir, device=device, **kwargs)

    return hf_pipeline
