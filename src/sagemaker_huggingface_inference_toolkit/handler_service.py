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

import importlib
import logging
import os
import time
import sys
from abc import ABC

from sagemaker_inference import environment
from transformers.pipelines import SUPPORTED_TASKS

from mms.service import PredictionException
from mms import metrics
from sagemaker_huggingface_inference_toolkit import decoder_encoder
from sagemaker_huggingface_inference_toolkit.transformers_utils import (
    _is_gpu_available,
    get_pipeline,
    infer_task_from_model_architecture,
)


ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"
PYTHON_PATH_ENV = "PYTHONPATH"

logger = logging.getLogger(__name__)


class HuggingFaceHandlerService(ABC):
    """Default handler service that is executed by the model server.

    The handler service is responsible for defining our InferenceHandler.
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.

    Implementation of: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """

    def __init__(self):
        self.error = None
        self.batch_size = 1
        self.model_dir = None
        self.model = None
        self.device = -1
        self.initialized = False
        self.context = None
        self.manifest = None
        self.environment = environment.Environment()

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.context = context
        properties = context.system_properties
        self.model_dir = properties.get("model_dir")
        self.batch_size = context.system_properties["batch_size"]

        code_dir_path = os.path.join(self.model_dir, "code")
        sys.path.insert(0, code_dir_path)
        self.validate_and_initialize_user_module()

        self.device = self.get_device()
        self.model = self.load(model_dir=self.model_dir)
        self.initialized = True
        # # Load methods from file
        # if (not self._initialized) and ENABLE_MULTI_MODEL:
        #     code_dir = os.path.join(context.system_properties.get("model_dir"), "code")
        #     sys.path.append(code_dir)
        #     self._initialized = True
        # # add model_dir/code to python path

    def get_device(self):
        """
        The get device function will return the device for the DL Framework.
        """
        if _is_gpu_available():
            return int(self.context.system_properties.get("gpu_id"))
        else:
            return -1

    def load(self, model_dir):
        """
        The Load handler is responsible for loading the Hugging Face transformer model.
        It can be overridden to load the model from storage
        Returns:
            hf_pipeline (Pipeline): A Hugging Face Transformer pipeline.
        """
        # gets pipeline from task tag
        if "HF_TASK" in os.environ:
            hf_pipeline = get_pipeline(task=os.environ["HF_TASK"], model_dir=model_dir, device=self.device)
        elif "config.json" in os.listdir(model_dir):
            task = infer_task_from_model_architecture(f"{model_dir}/config.json")
            hf_pipeline = get_pipeline(task=task, model_dir=model_dir, device=self.device)
        else:
            raise ValueError(
                f"You need to define one of the following {list(SUPPORTED_TASKS.keys())} as env 'TASK'.", 403
            )
        return hf_pipeline

    def preprocess(self, input_data):
        """
        The preprocess handler is responsible for deserializing the input data into
        an object for prediction, can handle JSON.
        The preprocess handler can be overridden for data or feature transformation,
        Args:
            input_data: the request payload serialized in the content_type format

        Returns:
            decoded_input_data (dict): deserialized input_data into a Python dictonary.
        """
        decoded_input_data = decoder_encoder.decode(input_data)
        return decoded_input_data

    def predict(self, data):
        """The predict handler is responsible for model predictions. Calls the `__call__` method of the provided `Pipeline`
        on decoded_input_data deserialized in input_fn. Runs prediction on GPU if is available.
        The predict handler can be overridden to implement the model inference.
        Args:
            data (dict): deserialized decoded_input_data returned by the input_fn
        Returns:
            obj (dict): prediction result.
        """

        # pop inputs for pipeline
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        start_time = time.time()
        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = self.model(inputs, **parameters)
        else:
            prediction = self.model(inputs)
        # add raw predict metric
        self.context.metrics.add_time("RawPredictTime", int(round((time.time() - start_time) * 1000)), None, "ms")

        # logger.info(f"inference time {time.time()-start_time}s")
        return prediction

    def postprocess(self, prediction):
        """
        The postprocess handler is responsible for serializing the prediction result to
        the desired accept type, can handle JSON.
        The postprocess handler can be overridden for inference response transformation
        Args:
            prediction (dict): a prediction result from predict
        Returns: output data serialized
        """
        return decoder_encoder.encode(prediction)

    def handle(self, data, context):
        """Handles an inference request with input data and makes a prediction.

        Args:
            data (obj): the request data.
            context (obj): metadata on the incoming request data.

        Returns:
            list[obj]: The return value from the Transformer.transform method,
                which is a serialized prediction result wrapped in a list if
                inference is successful. Otherwise returns an error message
                with the context set appropriately.

        """
        try:
            if not self.initialized:
                self.initialize(context)

            input_data = data[0].get("body").decode("utf-8")

            processed_data = self.preprocess(input_data)
            predictions = self.predict(processed_data)
            output = self.postprocess(predictions)
            return [output]
        except Exception as e:
            raise PredictionException(str(e), 400)

    def validate_and_initialize_user_module(self):
        """Retrieves and validates the inference handlers provided within the user module.
        Can override load, preprocess, predict and post process function.
        """
        user_module_name = self.environment.module_name
        if importlib.util.find_spec(user_module_name) is not None:
            user_module = importlib.import_module(user_module_name)

            load_fn = getattr(user_module, "load_fn", None)
            preprocess_fn = getattr(user_module, "preprocess_fn", None)
            predict_fn = getattr(user_module, "predict_fn", None)
            postprocess_fn = getattr(user_module, "postprocess_fn", None)

            if load_fn is not None:
                self.load = load_fn
            if preprocess_fn is not None:
                self.preprocess = preprocess_fn
            if predict_fn is not None:
                self.predict = predict_fn
            if postprocess_fn is not None:
                self.postprocess = postprocess_fn
