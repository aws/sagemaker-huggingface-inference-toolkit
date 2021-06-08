<!---
Copyright 2021 The HuggingFace Team, Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<div style="display:flex; text-align:center;">
<img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="100"/> 
<img src="https://github.com/aws/sagemaker-inference-toolkit/raw/master/branding/icon/sagemaker-banner.png" width="450"/>
</div>


# SageMaker Hugging Face Inference Toolkit 

[![Latest Version](https://img.shields.io/pypi/v/sagemaker_huggingface_inference_toolkit.svg)](https://pypi.python.org/pypi/sagemaker_huggingface_inference_toolkit) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker_huggingface_inference_toolkit.svg)](https://pypi.python.org/pypi/sagemaker_huggingface_inference_toolkit) [![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)


SageMaker Hugging Face Inference Toolkit is an open-source library for serving 🤗 Transformers models on Amazon SageMaker. This library provides default pre-processing, predict and postprocessing for certain 🤗 Transformers models and tasks. It utilizes the [SageMaker Inference Toolkit](https://github.com/aws/sagemaker-inference-toolkit) for starting up the model server, which is responsible for handling inference requests.

For Training, see [Run training on Amazon SageMaker](https://huggingface.co/transformers/sagemaker.html).

For the Dockerfiles used for building SageMaker Hugging Face Containers, see [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers/tree/master/huggingface).

For information on running Hugging Face jobs on Amazon SageMaker, please refer to the [🤗 Transformers documentation](https://huggingface.co/transformers/sagemaker.html).

For notebook examples: [SageMaker Notebook Examples](https://github.com/huggingface/notebooks/tree/master/sagemaker).

---
## 💻  Getting Started with 🤗 Inference Toolkit

_needs to be adjusted -> currently pseudo code_

**Install Amazon SageMaker Python SDK**

```bash
pip install sagemaker --upgrade
```

**Create a Amazon SageMaker endpoint with a trained model.**

```python
from sagemaker.huggingface import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.4',
    pytorch_version='1.6',
    model_data='s3://my-trained-model/artifcats/model.tar.gz',
    role=role,
)
# deploy model to SageMaker Inference
huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")
```


**Create a Amazon SageMaker endpoint with a model from the [🤗 Hub](https://huggingface.co/models).**

```python
from sagemaker.huggingface import HuggingFaceModel
# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad',
  'HF_TASK':'question-answering'
}
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.4',
    pytorch_version='1.6',
    env=hub,
    role=role,
    name=hub['HF_MODEL_ID'], 
)
# deploy model to SageMaker Inference
huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")
```

---

## 🛠️ Environment variables

The SageMaker Hugging Face Inference Toolkit implements various additional environment variables to simplify your deployment experience. A full list of environment variables is given below.

#### `HF_TASK`

The `HF_TASK` environment variable defines the task for the used 🤗 Transformers pipeline. A full list of tasks can be find [here](https://huggingface.co/transformers/main_classes/pipelines.html).

```bash
HF_TASK="question-answering"
```

#### `HF_MODEL_ID`

The `HF_MODEL_ID` environment variable defines the model id, which will be automatically loaded from [huggingface.co/models](https://huggingface.co/models) when creating or SageMaker Endpoint. The 🤗 Hub provides +10 000 models all available through this environment variable.

```bash
HF_MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english"
```

#### `HF_MODEL_REVISION`

The `HF_MODEL_REVISION` is an extension to `HF_MODEL_ID` and allows you to define/pin a revision of the model to make sure you always load the same model on your SageMaker Endpoint.

```bash
HF_MODEL_REVISION="03b4d196c19d0a73c7e0322684e97db1ec397613"
```

#### `HF_API_TOKEN`

The `HF_API_TOKEN` environment variable defines the your Hugging Face authorization token. The `HF_API_TOKEN` is used as a HTTP bearer authorization for remote files, like private models. You can find your token at your [settings page](https://huggingface.co/settings/token).

```bash
HF_API_TOKEN="api_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

---

## 🧑🏻‍💻 User defined code/modules

The Hugging Face Inference Toolkit allows user to override the default methods of the `HuggingFaceHandlerService`. Therefor the need to create a named `code/` with a `inference.py` file in it. 
For example:  
```bash
model.tar.gz/
|- pytroch_model.bin
|- ....
|- code/
  |- inference.py
  |- requirements.txt 
```
In this example, `pytroch_model.bin` is the model file saved from training, `inference.py` is the custom inference module, and `requirements.txt` is a requirements file to add additional dependencies.
The custom module can override the following methods:  

* `load_fn(model_dir)`: overrides the default method for loading the model, the return value `model` will be used in the `predict()` for predicitions. It receives argument the `model_dir`, where your `model.tar.gz` is saved.
* `preprocess_fn(input_data)`: overrides the default method for prerprocessing, the return value `data` will be used in the `predict()` method for predicitions. It receives the body of your request (raw).
* `predict(processed_data)`: overrides the default method for predictions, the return value `predictions` will be used in the `preprocess()` method.
* `postprocess(processed_data)`: overrides the default method for preprocessing, the return value `predictions` will be the respond of your request(e.g.`JSON`).




---
## 🤝 Contributing

Please read
`CONTRIBUTING.md <https://github.com/aws/sagemaker-pytorch-container/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us.

---
## 📜  License


SageMaker PyTorch Serving Container is licensed under the Apache 2.0 License. It is copyright 2018 Amazon
.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/