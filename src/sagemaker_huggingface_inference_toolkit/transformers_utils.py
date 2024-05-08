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
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login, snapshot_download
from transformers import AutoTokenizer, pipeline
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.pipelines import Conversation, Pipeline

from sagemaker_huggingface_inference_toolkit.diffusers_utils import get_diffusers_pipeline, is_diffusers_available
from sagemaker_huggingface_inference_toolkit.optimum_utils import (
    get_optimum_neuron_pipeline,
    is_optimum_neuron_available,
)


if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch

_aws_neuron_available = importlib.util.find_spec("torch_neuron") is not None


def is_aws_neuron_available():
    return _aws_neuron_available


def strtobool(val):
    """Convert a string representation of truth to True or False.
    True values are 'y', 'yes', 't', 'true', 'on', '1', 'TRUE', or 'True'; false values
    are 'n', 'no', 'f', 'false', 'off', '0', 'FALSE' or 'False.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1", "TRUE", "True"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0", "FALSE", "False"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


logger = logging.getLogger(__name__)


FRAMEWORK_MAPPING = {
    "pytorch": "pytorch*",
    "tensorflow": "tf*",
    "tf": "tf*",
    "pt": "pytorch*",
    "flax": "flax*",
    "rust": "rust*",
    "onnx": "*onnx*",
    "safetensors": "*safetensors",
    "coreml": "*mlmodel",
    "tflite": "*tflite",
    "savedmodel": "*tar.gz",
    "openvino": "*openvino*",
    "ckpt": "*ckpt",
    "neuronx": "*neuron",
}


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
TRUST_REMOTE_CODE = strtobool(os.environ.get("HF_TRUST_REMOTE_CODE", "False"))


def create_artifact_filter(framework):
    """
    Returns a list of regex pattern based on the DL Framework. which will be to used to ignore files when downloading
    """
    ignore_regex_list = list(set(FRAMEWORK_MAPPING.values()))

    pattern = FRAMEWORK_MAPPING.get(framework, None)
    if pattern in ignore_regex_list:
        ignore_regex_list.remove(pattern)
        return ignore_regex_list
    else:
        return []


def wrap_conversation_pipeline(pipeline):
    def wrapped_pipeline(inputs, *args, **kwargs):
        converted_input = Conversation(
            inputs["text"],
            past_user_inputs=inputs.get("past_user_inputs", []),
            generated_responses=inputs.get("generated_responses", []),
        )
        prediction = pipeline(converted_input, *args, **kwargs)
        return {
            "generated_text": prediction.generated_responses[-1],
            "conversation": {
                "past_user_inputs": prediction.past_user_inputs,
                "generated_responses": prediction.generated_responses,
            },
        }

    return wrapped_pipeline


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
    if use_auth_token is not None:
        login(token=use_auth_token)
    # extracts base framework
    framework = _get_framework()

    # creates directory for saved model based on revision and model
    storage_folder = _build_storage_path(model_id, model_dir, revision)
    os.makedirs(storage_folder, exist_ok=True)

    # check if safetensors weights are available
    if framework == "pytorch":
        files = HfApi().model_info(model_id).siblings
        if is_optimum_neuron_available() and any(f.rfilename.endswith("neuron") for f in files):
            framework = "neuronx"
        elif any(f.rfilename.endswith("safetensors") for f in files):
            framework = "safetensors"

    # create regex to only include the framework specific weights
    ignore_regex = create_artifact_filter(framework)

    # Download the repository to the workdir and filter out non-framework specific weights
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=str(storage_folder),
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_regex,
    )

    return storage_folder


def infer_task_from_model_architecture(model_config_path: str, architecture_index=0) -> str:
    """
    Infer task from `config.json` of trained model. It is not guaranteed to the detect, e.g. some models implement multiple architectures or
    trainend on different tasks https://huggingface.co/facebook/bart-large/blob/main/config.json. Should work for every on Amazon SageMaker fine-tuned model.
    It is always recommended to set the task through the env var `TASK`.
    """
    with open(model_config_path, "r") as config_file:
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
    # set env to work with
    os.environ["HF_TASK"] = task
    return task


def infer_task_from_hub(model_id: str, revision: Optional[str] = None, use_auth_token: Optional[str] = None) -> str:
    """
    Infer task from Hub by extracting `pipeline_tag` for model_info.
    """
    _api = HfApi()
    model_info = _api.model_info(repo_id=model_id, revision=revision, token=use_auth_token)
    if model_info.pipeline_tag is not None:
        # set env to work with
        os.environ["HF_TASK"] = model_info.pipeline_tag
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
    # define tokenizer or feature extractor as kwargs to load it the pipeline correctly
    if task in {
        "automatic-speech-recognition",
        "image-segmentation",
        "image-classification",
        "audio-classification",
        "object-detection",
        "zero-shot-image-classification",
    }:
        kwargs["feature_extractor"] = model_dir
    else:
        kwargs["tokenizer"] = model_dir
    # check if optimum neuron is available and tries to load it
    if is_optimum_neuron_available():
        hf_pipeline = get_optimum_neuron_pipeline(task=task, model_dir=model_dir)
    elif TRUST_REMOTE_CODE and os.environ.get("HF_MODEL_ID", None) is not None and device == 0:
        tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL_ID"])

        hf_pipeline = pipeline(
            task=task,
            model=os.environ["HF_MODEL_ID"],
            tokenizer=tokenizer,
            trust_remote_code=TRUST_REMOTE_CODE,
            model_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        )
    elif is_diffusers_available() and task == "text-to-image":
        hf_pipeline = get_diffusers_pipeline(task=task, model_dir=model_dir, device=device, **kwargs)
    else:
        # load pipeline
        hf_pipeline = pipeline(
            task=task, model=model_dir, device=device, trust_remote_code=TRUST_REMOTE_CODE, **kwargs
        )

    # wrapp specific pipeline to support better ux
    if task == "conversational":
        hf_pipeline = wrap_conversation_pipeline(hf_pipeline)

    return hf_pipeline
