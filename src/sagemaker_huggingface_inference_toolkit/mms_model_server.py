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

import os
import pathlib
import subprocess

from sagemaker_inference import environment, logging
from sagemaker_inference.environment import model_dir
from sagemaker_inference.model_server import (
    DEFAULT_MMS_LOG_FILE,
    ENABLE_MULTI_MODEL,
    MMS_CONFIG_FILE,
    REQUIREMENTS_PATH,
    _add_sigchild_handler,
    _add_sigterm_handler,
    _create_model_server_config_file,
    _install_requirements,
    _retry_retrieve_mms_server_process,
    _set_python_path,
)

from sagemaker_huggingface_inference_toolkit import handler_service
from sagemaker_huggingface_inference_toolkit.transformers_utils import (
    HF_API_TOKEN,
    HF_MODEL_REVISION,
    _load_model_from_hub,
    is_aws_neuron_available,
)


logger = logging.get_logger()

DEFAULT_HANDLER_SERVICE = handler_service.__name__

DEFAULT_MMS_MODEL_DIRECTORY = os.path.join(os.getcwd(), ".sagemaker/mms/models")
MODEL_STORE = "/" if ENABLE_MULTI_MODEL else DEFAULT_MMS_MODEL_DIRECTORY


def start_model_server(handler_service=DEFAULT_HANDLER_SERVICE):
    """Configure and start the model server.

    Args:
        handler_service (str): python path pointing to a module that defines
            a class with the following:

                - A ``handle`` method, which is invoked for all incoming inference
                    requests to the model server.
                - A ``initialize`` method, which is invoked at model server start up
                    for loading the model.

            Defaults to ``sagemaker_huggingface_inference_toolkit.handler_service``.

    """
    if ENABLE_MULTI_MODEL:
        if not os.getenv("SAGEMAKER_HANDLER"):
            os.environ["SAGEMAKER_HANDLER"] = handler_service
        _set_python_path()
    elif "HF_MODEL_ID" in os.environ:
        if is_aws_neuron_available():
            raise ValueError(
                "Hugging Face Hub deployments are currently not supported with AWS Neuron and Inferentia."
                "You need to create a `inference.py` script to run your model using AWS Neuron"
            )
        storage_dir = _load_model_from_hub(
            model_id=os.environ["HF_MODEL_ID"],
            model_dir=DEFAULT_MMS_MODEL_DIRECTORY,
            revision=HF_MODEL_REVISION,
            use_auth_token=HF_API_TOKEN,
        )
        _adapt_to_mms_format(handler_service, storage_dir)
    else:
        _adapt_to_mms_format(handler_service, model_dir)

    env = environment.Environment()
    _create_model_server_config_file(env)

    if os.path.exists(REQUIREMENTS_PATH):
        _install_requirements()

    multi_model_server_cmd = [
        "multi-model-server",
        "--start",
        "--model-store",
        MODEL_STORE,
        "--mms-config",
        MMS_CONFIG_FILE,
        "--log-config",
        DEFAULT_MMS_LOG_FILE,
    ]

    logger.info(multi_model_server_cmd)
    subprocess.Popen(multi_model_server_cmd)
    # retry for configured timeout
    mms_process = _retry_retrieve_mms_server_process(env.startup_timeout)

    _add_sigterm_handler(mms_process)
    _add_sigchild_handler()

    mms_process.wait()


def _adapt_to_mms_format(handler_service, model_path):
    os.makedirs(DEFAULT_MMS_MODEL_DIRECTORY, exist_ok=True)

    # gets the model from the path, default is model/
    model = pathlib.PurePath(model_path)

    # This is archiving or cp /opt/ml/model to /opt/ml (MODEL_STORE) into model (MODEL_NAME)
    model_archiver_cmd = [
        "model-archiver",
        "--model-name",
        model.name,
        "--handler",
        handler_service,
        "--model-path",
        model_path,
        "--export-path",
        DEFAULT_MMS_MODEL_DIRECTORY,
        "--archive-format",
        "no-archive",
        "--f",
    ]

    logger.info(model_archiver_cmd)
    subprocess.check_call(model_archiver_cmd)

    _set_python_path()
