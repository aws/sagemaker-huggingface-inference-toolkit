<div style="display:flex; text-align:center;">
<img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="100"/> 
<img src="https://github.com/aws/sagemaker-inference-toolkit/raw/master/branding/icon/sagemaker-banner.png" width="450"/>
</div>




# SageMaker Hugging Face Inference Toolkit 

[![Latest Version](https://img.shields.io/pypi/v/sagemaker_huggingface_inference_toolkit.svg)](https://pypi.python.org/pypi/sagemaker_huggingface_inference_toolkit) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/sagemaker_huggingface_inference_toolkit.svg)](https://pypi.python.org/pypi/sagemaker_huggingface_inference_toolkit) [![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)


SageMaker Hugging Face Inference Toolkit is an open-source library for serving 🤗 Transformers and Diffusers models on Amazon SageMaker. This library provides default pre-processing, predict and postprocessing for certain 🤗 Transformers and Diffusers models and tasks. It utilizes the [SageMaker Inference Toolkit](https://github.com/aws/sagemaker-inference-toolkit) for starting up the model server, which is responsible for handling inference requests.

For Training, see [Run training on Amazon SageMaker](https://huggingface.co/docs/sagemaker/train).

For the Dockerfiles used for building SageMaker Hugging Face Containers, see [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers/tree/master/huggingface).

For information on running Hugging Face jobs on Amazon SageMaker, please refer to the [🤗 Transformers documentation](https://huggingface.co/docs/sagemaker).

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
    transformers_version='4.6',
    pytorch_version='1.7',
    py_version='py36',
    model_data='s3://my-trained-model/artifacts/model.tar.gz',
    role=role,
)
# deploy model to SageMaker Inference
huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")
```


**Create a Amazon SageMaker endpoint with a model from the [🤗 Hub](https://huggingface.co/models).**  
_note: This is an experimental feature, where the model will be loaded after the endpoint is created. Not all sagemaker features are supported, e.g. MME_
```python
from sagemaker.huggingface import HuggingFaceModel
# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad',
  'HF_TASK':'question-answering'
}
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.6',
    pytorch_version='1.7',
    py_version='py36',
    env=hub,
    role=role,
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

#### `HF_TRUST_REMOTE_CODE`

The `HF_TRUST_REMOTE_CODE` environment variable defines wether or not to allow for custom models defined on the Hub in their own modeling files. Allowed values are `"True"` and `"False"`

```bash
HF_TRUST_REMOTE_CODE="True"
```

#### `HF_OPTIMUM_BATCH_SIZE`

The `HF_OPTIMUM_BATCH_SIZE` environment variable defines the batch size, which is used when compiling the model to Neuron. The default value is `1`. Not required when model is already converted. 

```bash
HF_OPTIMUM_BATCH_SIZE="1"
```

#### `HF_OPTIMUM_SEQUENCE_LENGTH`

The `HF_OPTIMUM_SEQUENCE_LENGTH` environment variable defines the sequence length, which is used when compiling the model to Neuron. There is no default value. Not required when model is already converted. 

```bash
HF_OPTIMUM_SEQUENCE_LENGTH="128"
```

---

## 🧑🏻‍💻 User defined code/modules

The Hugging Face Inference Toolkit allows user to override the default methods of the `HuggingFaceHandlerService`. Therefore, they need to create a folder named `code/` with an `inference.py` file in it. You can find an example for it in [sagemaker/17_customer_inference_script](https://github.com/huggingface/notebooks/blob/master/sagemaker/17_custom_inference_script/sagemaker-notebook.ipynb).
For example:  
```bash
model.tar.gz/
|- pytorch_model.bin
|- ....
|- code/
  |- inference.py
  |- requirements.txt 
```
In this example, `pytorch_model.bin` is the model file saved from training, `inference.py` is the custom inference module, and `requirements.txt` is a requirements file to add additional dependencies.
The custom module can override the following methods:  

* `model_fn(model_dir)`: overrides the default method for loading the model, the return value `model` will be used in the `predict()` for predicitions. It receives argument the `model_dir`, the path to your unzipped `model.tar.gz`.
* `transform_fn(model, data, content_type, accept_type)`: overrides the default transform function with a custom implementation. Customers using this would have to implement `preprocess`, `predict` and `postprocess` steps in the `transform_fn`. **NOTE: This method can't be combined with `input_fn`, `predict_fn` or `output_fn` mentioned below.** 
* `input_fn(input_data, content_type)`: overrides the default method for preprocessing, the return value `data` will be used in the `predict()` method for predicitions. The input is `input_data`, the raw body of your request and `content_type`, the content type form the request Header.
* `predict_fn(processed_data, model)`: overrides the default method for predictions, the return value `predictions` will be used in the `postprocess()` method. The input is `processed_data`, the result of the `preprocess()` method.
* `output_fn(prediction, accept)`: overrides the default method for postprocessing, the return value `result` will be the respond of your request(e.g.`JSON`). The inputs are `predictions`, the result of the `predict()` method and `accept` the return accept type from the HTTP Request, e.g. `application/json`


## 🏎️ Deploy Models on AWS Inferentia2

The SageMaker Hugging Face Inference Toolkit provides support for deploying Hugging Face on AWS Inferentia2. To deploy a model on Inferentia2 you have 3 options:
* Provide an already compiled model with a `model.neuron` file as `HF_MODEL_ID`, .e.g. `optimum/tiny_random_bert_neuron`
* Provide the `HF_OPTIMUM_BATCH_SIZE` and `HF_OPTIMUM_SEQUENCE_LENGTH` environment variables to compile the model on the fly, e.g. `HF_OPTIMUM_BATCH_SIZE=1 HF_OPTIMUM_SEQUENCE_LENGTH=128`
* Include `neuron` dictionary in the [config.json](https://huggingface.co/optimum/tiny_random_bert_neuron/blob/main/config.json) file in the model archive, e.g. `neuron: {"static_batch_size": 1, "static_sequence_length": 128}`

The currently supported tasks can be found [here](https://huggingface.co/docs/optimum-neuron/en/package_reference/supported_models). If you plan to deploy an LLM, we recommend taking a look at [Neuronx TGI](https://huggingface.co/blog/text-generation-inference-on-inferentia2), which is purposly build for LLMs 

---
## 🤝 Contributing

Please read [CONTRIBUTING.md](https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/CONTRIBUTING.md)
for details on our code of conduct, and the process for submitting pull
requests to us.

---
## 📜  License

SageMaker Hugging Face Inference Toolkit is licensed under the Apache 2.0 License.

---

## 🧑🏻‍💻 Development Environment

Install all test and development packages with 

```bash
pip3 install -e ".[test,dev]"
```
## Run Model Locally

1. manually change `MMS_CONFIG_FILE`
```
wget -O sagemaker-mms.properties https://raw.githubusercontent.com/aws/deep-learning-containers/master/huggingface/build_artifacts/inference/config.properties
```

2. Run Container, e.g. `text-to-image`
```
HF_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0" HF_TASK="text-to-image" python src/sagemaker_huggingface_inference_toolkit/serving.py
```
3. Adjust `handler_service.py` and comment out `if content_type in content_types.UTF8_TYPES:` thats needed for SageMaker but cannot be used locally

3. Send request 
```
curl --request POST \
  --url http://localhost:8080/invocations \
  --header 'Accept: image/png' \
  --header 'Content-Type: application/json' \
  --data '"{\"inputs\": \"Camera\"}" \
  --output image.png
```


## Run Inferentia2 Model Locally

_Note: You need to run this on an Inferentia2 instance._

1. manually change `MMS_CONFIG_FILE`
```
wget -O sagemaker-mms.properties https://raw.githubusercontent.com/aws/deep-learning-containers/master/huggingface/build_artifacts/inference/config.properties
```

2. Adjust `handler_service.py` and comment out `if content_type in content_types.UTF8_TYPES:` thats needed for SageMaker but cannot be used locally

2. Run Container,

- transformers `text-classification` with `HF_OPTIMUM_BATCH_SIZE` and `HF_OPTIMUM_SEQUENCE_LENGTH`
```
HF_MODEL_ID="distilbert/distilbert-base-uncased-finetuned-sst-2-english" HF_TASK="text-classification" HF_OPTIMUM_BATCH_SIZE=1 HF_OPTIMUM_SEQUENCE_LENGTH=128 python src/sagemaker_huggingface_inference_toolkit/serving.py
```
- sentence transformers `feature-extration` with `HF_OPTIMUM_BATCH_SIZE` and `HF_OPTIMUM_SEQUENCE_LENGTH`
```
HF_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2" HF_TASK="feature-extraction" HF_OPTIMUM_BATCH_SIZE=1 HF_OPTIMUM_SEQUENCE_LENGTH=128 python src/sagemaker_huggingface_inference_toolkit/serving.py
```

3. Send request 
```
curl --request POST \
  --url http://localhost:8080/invocations \
  --header 'Content-Type: application/json' \
  --data "{\"inputs\": \"I like you.\"}"
```