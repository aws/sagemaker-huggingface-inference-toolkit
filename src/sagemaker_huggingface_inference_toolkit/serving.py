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
from subprocess import CalledProcessError

from retrying import retry

from sagemaker_huggingface_inference_toolkit import handler_service, mms_model_server


HANDLER_SERVICE = handler_service.__name__


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_mms():
    mms_model_server.start_model_server(handler_service=HANDLER_SERVICE)


def main():
    _start_mms()


if __name__ == "__main__":
    main()
