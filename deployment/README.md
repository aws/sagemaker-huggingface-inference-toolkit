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
# Deployment

## Environment Configuration

pass as ENV `HF_TASKS` with the tasks that should be used and preferable `HF_MODEL_ID` if you want to use a model from huggingface.co. 

https://sagemaker.readthedocs.io/en/stable/api/inference/model.html

```python
huggingface_model = Model(
  env={'HF_TASK':'text-classification',
      'HF_MODEL_ID':'bert-base-uncased' # if you want to use a model from the hub
      }
  )
```

## Create SageMaker Endpoint

### Deploy model from the HF hub

#### Pytorch

```bash
# make endpoint
python ./deployment/deploy.py \
      --image_uri 558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch:cpu-0.0.1  \
      --model_id distilbert-base-uncased-distilled-squad \
      --hf_task question-answering \
      --instance_type ml.m5.large
```

##### Tensorflow

```bash
# make endpoint
python ./deployment/deploy.py \
      --image_uri 558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-tensorflow:cpu-0.0.1  \
      --model_id dbmdz/bert-large-cased-finetuned-conll03-english \
      --hf_task ner \
      --instance_type ml.m5.xlarge
```

### Deploy S3 stored model

#### With task definition

```bash
# make endpoint
python ./deployment/deploy.py \
      --image_uri 558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch:cpu-0.0.1  \
      --model_data s3://sagemaker-us-east-1-558105141721/huggingface-pytorch-training-2021-03-30-19-47-53-063/output/model.tar.gz \
      --hf_task summarization \
      --name bart-sum \
      --instance_type ml.m5.2xlarge
```

#### Without task definition

```bash
# make endpoint
python ./deployment/deploy.py \
      --image_uri 558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch:cpu-0.0.1  \
      --model_data s3://sagemaker-us-east-1-558105141721/huggingface-pytorch-training-2021-03-30-19-47-53-063/output/model.tar.gz \
      --name bart-sum \
      --instance_type ml.m5.2xlarge
```


### Code behind `deploy.py`

```python
hf_model = Model(
    image_uri=image_uri,  # A Docker image URI.
    model_data=model_data,  # The S3 location of a SageMaker model data .tar.gz
    env=env,  # Environment variables to run with image_uri when hosted in SageMaker (default: None).
    role=role_name,  # An AWS IAM role (either name or full ARN).
    name=name,  # The model name
    sagemaker_session=sess,
)

hf_model.deploy(
    initial_instance_count=deploy_config["initial_instance_count"],
    instance_type=deploy_config["instance_type"],
    endpoint_name=name,
)
```

Choose the right API/SDK to invoke your endpoint. https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html