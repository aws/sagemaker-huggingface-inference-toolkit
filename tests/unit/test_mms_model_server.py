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
# limitations under the License.import os
import os
from sagemaker_inference.environment import model_dir

from mock import patch
from sagemaker_huggingface_inference_toolkit import mms_model_server, transformers_utils


PYTHON_PATH = "python_path"
DEFAULT_CONFIGURATION = "default_configuration"


@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._retry_retrieve_mms_server_process")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._add_sigterm_handler")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._install_requirements")
@patch("os.path.exists", return_value=True)
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._create_model_server_config_file")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._adapt_to_mms_format")
@patch("sagemaker_inference.environment.Environment")
def test_start_mms_default_service_handler(
    env,
    adapt,
    create_config,
    exists,
    install_requirements,
    sigterm,
    retrieve,
    subprocess_popen,
    subprocess_call,
):
    env.return_value.startup_timeout = 10000
    mms_model_server.start_model_server()

    # In this case, we should not rearchive the model
    adapt.assert_not_called()

    create_config.assert_called_once_with(env.return_value, mms_model_server.DEFAULT_HANDLER_SERVICE)
    exists.assert_called_once_with(mms_model_server.REQUIREMENTS_PATH)
    install_requirements.assert_called_once_with()

    multi_model_server_cmd = [
        "multi-model-server",
        "--start",
        "--model-store",
        mms_model_server.DEFAULT_MODEL_STORE,
        "--mms-config",
        mms_model_server.MMS_CONFIG_FILE,
        "--log-config",
        mms_model_server.DEFAULT_MMS_LOG_FILE,
        "--models",
        "{}={}".format(mms_model_server.DEFAULT_MMS_MODEL_NAME, model_dir),
    ]

    subprocess_popen.assert_called_once_with(multi_model_server_cmd)
    sigterm.assert_called_once_with(retrieve.return_value)


@patch("sagemaker_huggingface_inference_toolkit.transformers_utils._aws_neuron_available", return_value=True)
@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._retry_retrieve_mms_server_process")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._load_model_from_hub")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._add_sigterm_handler")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._install_requirements")
@patch("os.makedirs", return_value=True)
@patch("os.remove", return_value=True)
@patch("os.path.exists", return_value=True)
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._create_model_server_config_file")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._adapt_to_mms_format")
@patch("sagemaker_inference.environment.Environment")
def test_start_mms_neuron(
    env,
    adapt,
    create_config,
    exists,
    remove,
    dir,
    install_requirements,
    sigterm,
    load_model_from_hub,
    retrieve,
    subprocess_popen,
    subprocess_call,
    is_aws_neuron_available,
):
    env.return_value.startup_timeout = 10000
    mms_model_server.start_model_server()

    # In this case, we should not call model archiver
    adapt.assert_not_called()

    create_config.assert_called_once_with(env.return_value, mms_model_server.DEFAULT_HANDLER_SERVICE)
    exists.assert_called_once_with(mms_model_server.REQUIREMENTS_PATH)
    install_requirements.assert_called_once_with()

    multi_model_server_cmd = [
        "multi-model-server",
        "--start",
        "--model-store",
        mms_model_server.DEFAULT_MODEL_STORE,
        "--mms-config",
        mms_model_server.MMS_CONFIG_FILE,
        "--log-config",
        mms_model_server.DEFAULT_MMS_LOG_FILE,
        "--models",
        "{}={}".format(mms_model_server.DEFAULT_MMS_MODEL_NAME, model_dir),
    ]

    subprocess_popen.assert_called_once_with(multi_model_server_cmd)
    sigterm.assert_called_once_with(retrieve.return_value)


@patch("subprocess.call")
@patch("subprocess.Popen")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._retry_retrieve_mms_server_process")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._load_model_from_hub")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._add_sigterm_handler")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._install_requirements")
@patch("os.makedirs", return_value=True)
@patch("os.remove", return_value=True)
@patch("os.path.exists", return_value=True)
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._create_model_server_config_file")
@patch("sagemaker_huggingface_inference_toolkit.mms_model_server._adapt_to_mms_format")
@patch("sagemaker_inference.environment.Environment")
def test_start_mms_with_model_from_hub(
    env,
    adapt,
    create_config,
    exists,
    remove,
    dir,
    install_requirements,
    sigterm,
    load_model_from_hub,
    retrieve,
    subprocess_popen,
    subprocess_call,
):
    env.return_value.startup_timeout = 10000

    os.environ["HF_MODEL_ID"] = "lysandre/tiny-bert-random"

    mms_model_server.start_model_server()

    load_model_from_hub.assert_called_once_with(
        model_id=os.environ["HF_MODEL_ID"],
        model_dir=mms_model_server.DEFAULT_HF_HUB_MODEL_EXPORT_DIRECTORY,
        revision=transformers_utils.HF_MODEL_REVISION,
        use_auth_token=transformers_utils.HF_API_TOKEN,
    )

    # When loading model from hub, we do call model archiver
    adapt.assert_called_once_with(mms_model_server.DEFAULT_HANDLER_SERVICE, load_model_from_hub())

    create_config.assert_called_once_with(env.return_value, mms_model_server.DEFAULT_HANDLER_SERVICE)
    exists.assert_called_with(mms_model_server.REQUIREMENTS_PATH)
    install_requirements.assert_called_once_with()

    multi_model_server_cmd = [
        "multi-model-server",
        "--start",
        "--model-store",
        mms_model_server.DEFAULT_HF_HUB_MODEL_EXPORT_DIRECTORY,
        "--mms-config",
        mms_model_server.MMS_CONFIG_FILE,
        "--log-config",
        mms_model_server.DEFAULT_MMS_LOG_FILE,
    ]

    subprocess_popen.assert_called_once_with(multi_model_server_cmd)
    sigterm.assert_called_once_with(retrieve.return_value)
    os.remove(mms_model_server.DEFAULT_HF_HUB_MODEL_EXPORT_DIRECTORY)
