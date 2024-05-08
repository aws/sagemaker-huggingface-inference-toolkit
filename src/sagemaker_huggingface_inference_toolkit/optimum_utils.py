# Copyright 2023 The HuggingFace Team, Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging
import os


_optimum_neuron = False
if importlib.util.find_spec("optimum") is not None:
    if importlib.util.find_spec("optimum.neuron") is not None:
        _optimum_neuron = True

logger = logging.getLogger(__name__)


def is_optimum_neuron_available():
    return _optimum_neuron


def get_input_shapes(model_dir):
    """Method to get input shapes from model config file. If config file is not present, default values are returned."""
    from transformers import AutoConfig

    input_shapes = {}
    input_shapes_available = False
    # try to get input shapes from config file
    try:
        config = AutoConfig.from_pretrained(model_dir)
        if hasattr(config, "neuron"):
            # check if static batch size and sequence length are available
            if config.neuron.get("static_batch_size", None) and config.neuron.get("static_sequence_length", None):
                input_shapes["batch_size"] = config.neuron["static_batch_size"]
                input_shapes["sequence_length"] = config.neuron["static_sequence_length"]
                input_shapes_available = True
                logger.info(
                    f"Input shapes found in config file. Using input shapes from config with batch size {input_shapes['batch_size']} and sequence length {input_shapes['sequence_length']}"
                )
            else:
                # Add warning if environment variables are set but will be ignored
                if os.environ.get("HF_OPTIMUM_BATCH_SIZE", None) is not None:
                    logger.warning(
                        "HF_OPTIMUM_BATCH_SIZE environment variable is set. Environment variable will be ignored and input shapes from config file will be used."
                    )
                if os.environ.get("HF_OPTIMUM_SEQUENCE_LENGTH", None) is not None:
                    logger.warning(
                        "HF_OPTIMUM_SEQUENCE_LENGTH environment variable is set. Environment variable will be ignored and input shapes from config file will be used."
                    )
    except Exception:
        input_shapes_available = False

    # return input shapes if available
    if input_shapes_available:
        return input_shapes

    # extract input shapes from environment variables
    sequence_length = os.environ.get("HF_OPTIMUM_SEQUENCE_LENGTH", None)
    if sequence_length is None:
        raise ValueError(
            "HF_OPTIMUM_SEQUENCE_LENGTH environment variable is not set. Please set HF_OPTIMUM_SEQUENCE_LENGTH to a positive integer."
        )

    if not int(sequence_length) > 0:
        raise ValueError(
            f"HF_OPTIMUM_SEQUENCE_LENGTH must be set to a positive integer. Current value is {sequence_length}"
        )
    batch_size = os.environ.get("HF_OPTIMUM_BATCH_SIZE", 1)
    logger.info(
        f"Using input shapes from environment variables with batch size {batch_size} and sequence length {sequence_length}"
    )
    return {"batch_size": int(batch_size), "sequence_length": int(sequence_length)}


def get_optimum_neuron_pipeline(task, model_dir):
    """Method to get optimum neuron pipeline for a given task. Method checks if task is supported by optimum neuron and if required environment variables are set, in case model is not converted. If all checks pass, optimum neuron pipeline is returned. If checks fail, an error is raised."""
    from optimum.neuron.pipelines.transformers.base import NEURONX_SUPPORTED_TASKS, pipeline
    from optimum.neuron.utils import NEURON_FILE_NAME

    # check task support
    if task not in NEURONX_SUPPORTED_TASKS:
        raise ValueError(
            f"Task {task} is not supported by optimum neuron and inf2. Supported tasks are: {list(NEURONX_SUPPORTED_TASKS.keys())}"
        )

    # check if model is already converted and has input shapes available
    export = True
    if NEURON_FILE_NAME in os.listdir(model_dir):
        export = False
    if export:
        logger.info("Model is not converted. Checking if required environment variables are set and converting model.")

    # get static input shapes to run inference
    input_shapes = get_input_shapes(model_dir)
    # set NEURON_RT_NUM_CORES to 1 to avoid conflicts with multiple HTTP workers
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    # get optimum neuron pipeline
    neuron_pipe = pipeline(task, model=model_dir, export=export, input_shapes=input_shapes)

    return neuron_pipe
