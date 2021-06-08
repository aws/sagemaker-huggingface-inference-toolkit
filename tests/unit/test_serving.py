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

from mock import patch


@patch("sagemaker_huggingface_inference_toolkit.mms_model_server.start_model_server")
def test_hosting_start(start_model_server):
    from sagemaker_huggingface_inference_toolkit import serving

    serving.main()
    start_model_server.assert_called_with(handler_service="sagemaker_huggingface_inference_toolkit.handler_service")


def test_retry_if_error():
    from sagemaker_huggingface_inference_toolkit import serving

    serving._retry_if_error(Exception)
