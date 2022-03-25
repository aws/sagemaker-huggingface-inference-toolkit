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
import sys
import time
from abc import ABC

from sagemaker_inference import environment, utils
from transformers.pipelines import SUPPORTED_TASKS

from mms.service import PredictionException
from sagemaker_huggingface_inference_toolkit import content_types, decoder_encoder
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
        self.model = self.load(self.model_dir)
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
                f"You need to define one of the following {list(SUPPORTED_TASKS.keys())} as env 'HF_TASK'.", 403
            )
        return hf_pipeline

    def preprocess(self, input_data, content_type):
        """
        The preprocess handler is responsible for deserializing the input data into
        an object for prediction, can handle JSON.
        The preprocess handler can be overridden for data or feature transformation,
        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns:
            decoded_input_data (dict): deserialized input_data into a Python dictonary.
        """
        # raises en error when using zero-shot-classification or table-question-answering, not possible due to nested properties
        if (
            os.environ.get("HF_TASK", None) == "zero-shot-classification"
            or os.environ.get("HF_TASK", None) == "table-question-answering"
        ) and content_type == content_types.CSV:
            raise PredictionException(
                f"content type {content_type} not support with {os.environ.get('HF_TASK', 'unknown task')}, use different content_type",
                400,
            )

        decoded_input_data = decoder_encoder.decode(input_data, content_type)
        return decoded_input_data

    def predict(self, data, model):
        """The predict handler is responsible for model predictions. Calls the `__call__` method of the provided `Pipeline`
        on decoded_input_data deserialized in input_fn. Runs prediction on GPU if is available.
        The predict handler can be overridden to implement the model inference.
        Args:
            data (dict): deserialized decoded_input_data returned by the input_fn
            model : Model returned by the `load` method or if it is a custom module `model_fn`.
        Returns:
            obj (dict): prediction result.
        """

        # pop inputs for pipeline
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = model(inputs, **parameters)
        else:
            prediction = model(inputs)
        return prediction

    def postprocess(self, prediction, accept):
        """
        The postprocess handler is responsible for serializing the prediction result to
        the desired accept type, can handle JSON.
        The postprocess handler can be overridden for inference response transformation
        Args:
            prediction (dict): a prediction result from predict
            accept (str): type which the output data needs to be serialized
        Returns: output data serialized
        """
        return decoder_encoder.encode(prediction, accept)

    def transform_fn(self, model, input_data, content_type, accept):
        """
        Transform function ("transform_fn") can be used to write one function with pre/post-processing steps and predict step in it.
        This fuction can't be mixed with "input_fn", "output_fn" or "predict_fn"
        Args:
            model: Model returned by the model_fn above
            input_data: Data received for inference
            content_type: The content type of the inference data
            accept: The response accept type.

        Returns: Response in the "accept" format type.

        """
        # run pipeline
        start_time = time.time()
        processed_data = self.preprocess(input_data, content_type)
        preprocess_time = time.time() - start_time
        predictions = self.predict(processed_data, model)
        predict_time = time.time() - preprocess_time - start_time
        response = self.postprocess(predictions, accept)
        postprocess_time = time.time() - predict_time - preprocess_time - start_time

        logger.info(
            f"Preprocess time - {preprocess_time * 1000} ms\n"
            f"Predict time - {predict_time * 1000} ms\n"
            f"Postprocess time - {postprocess_time * 1000} ms"
        )

        return response

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

            input_data = data[0].get("body")

            request_property = context.request_processor[0].get_request_properties()
            content_type = utils.retrieve_content_type_header(request_property)
            accept = request_property.get("Accept") or request_property.get("accept")

            if not accept or accept == content_types.ANY:
                accept = content_types.JSON

            if content_type in content_types.UTF8_TYPES:
                input_data = input_data.decode("utf-8")

            predict_start = time.time()
            response = self.transform_fn(self.model, input_data, content_type, accept)
            predict_end = time.time()

            context.metrics.add_time("Transform Fn", round((predict_end - predict_start) * 1000, 2))

            context.set_response_content_type(0, accept)
            return [response]

        except Exception as e:
            raise PredictionException(str(e), 400)

    def validate_and_initialize_user_module(self):
        """Retrieves and validates the inference handlers provided within the user module.
        Can override load, preprocess, predict and post process function.
        """
        user_module_name = self.environment.module_name
        if importlib.util.find_spec(user_module_name) is not None:
            user_module = importlib.import_module(user_module_name)

            load_fn = getattr(user_module, "model_fn", None)
            preprocess_fn = getattr(user_module, "input_fn", None)
            predict_fn = getattr(user_module, "predict_fn", None)
            postprocess_fn = getattr(user_module, "output_fn", None)
            transform_fn = getattr(user_module, "transform_fn", None)

            if transform_fn and (preprocess_fn or predict_fn or postprocess_fn):
                raise ValueError(
                    "Cannot use transform_fn implementation in conjunction with "
                    "input_fn, predict_fn, and/or output_fn implementation"
                )

            if load_fn is not None:
                self.load = load_fn
            if preprocess_fn is not None:
                self.preprocess = preprocess_fn
            if predict_fn is not None:
                self.predict = predict_fn
            if postprocess_fn is not None:
                self.postprocess = postprocess_fn
            if transform_fn is not None:
                self.transform_fn = transform_fn
