import json
import os

import numpy as np
import pytest

import boto3
from integ.config import task2input, task2model, task2output, task2performance, task2validation
from integ.utils import clean_up, count_tokens, timeout_and_delete_by_name, track_infer_time
from sagemaker import Session
from sagemaker.model import Model


ROLE_NAME = "sagemaker_execution_role"
PROFILE = "hf-sm"
REGION = "us-east-1"
os.environ["AWS_PROFILE"] = PROFILE  # setting aws profile for our boto3 client
os.environ["AWS_DEFAULT_REGION"] = REGION  # current DLCs are only in us-east-1 available

# TODO: Replace with released DLC images
images = {
    "pytorch": {
        "cpu": "558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch:cpu-0.0.1",
        "gpu": "558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch:gpu-0.0.1",
    },
    "tensorflow": {
        "cpu": "558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-tensorflow:cpu-0.0.1",
        "gpu": "558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-tensorflow:gpu-0.0.1",
    },
}


@pytest.mark.parametrize(
    "task",
    [
        "text-classification",
        "zero-shot-classification",
        "ner",
        "question-answering",
        "fill-mask",
        "summarization",
        "translation_xx_to_yy",
        "text2text-generation",
        "text-generation",
        "feature-extraction",
    ],
)
@pytest.mark.parametrize(
    "framework",
    [
        "pytorch",
    ],
)  # "tensorflow"])
@pytest.mark.parametrize(
    "device",
    [
        # "gpu",
        "cpu",
    ],
)
def test_deployment_from_hub(task, device, framework):
    image_uri = images[framework][device]
    name = f"hf-test-{framework}-{device}-{task}".replace("_", "-")
    model = task2model[task][framework]
    instance_type = "ml.m5.large" if device == "cpu" else "ml.g4dn.xlarge"
    number_of_requests = 100
    if model is None:
        return

    env = {"HF_MODEL_ID": model, "HF_TASK": task}

    sagemaker_session = Session()
    client = boto3.client("sagemaker-runtime")

    hf_model = Model(
        image_uri=image_uri,  # A Docker image URI.
        model_data=None,  # The S3 location of a SageMaker model data .tar.gz
        env=env,  # Environment variables to run with image_uri when hosted in SageMaker (default: None).
        role=ROLE_NAME,  # An AWS IAM role (either name or full ARN).
        name=name,  # The model name
        sagemaker_session=sagemaker_session,
    )

    with timeout_and_delete_by_name(name, sagemaker_session, minutes=59):
        # Use accelerator type to differentiate EI vs. CPU and GPU. Don't use processor value
        hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=name,
        )

        # Keep track of the inference time
        time_buffer = []

        # Warm up the model
        response = client.invoke_endpoint(
            EndpointName=name,
            Body=json.dumps(task2input[task]),
            ContentType="application/json",
            Accept="application/json",
        )

        # validate response
        response_body = response["Body"].read().decode("utf-8")

        assert True is task2validation[task](result=json.loads(response_body), snapshot=task2output[task])

        for _ in range(number_of_requests):
            with track_infer_time(time_buffer):
                response = client.invoke_endpoint(
                    EndpointName=name,
                    Body=json.dumps(task2input[task]),
                    ContentType="application/json",
                    Accept="application/json",
                )
        with open(f"{name}.json", "w") as outfile:
            data = {
                "index": name,
                "framework": framework,
                "device": device,
                "model": model,
                "number_of_requests": number_of_requests,
                "number_of_input_token": count_tokens(task2input[task]["inputs"], task),
                "average_request_time": np.mean(time_buffer),
                "max_request_time": max(time_buffer),
                "min_request_time": min(time_buffer),
                "p95_request_time": np.percentile(time_buffer, 95),
                "body": json.loads(response_body),
            }
            json.dump(data, outfile)

        assert task2performance[task][device]["average_request_time"] >= np.mean(time_buffer)

        clean_up(endpoint_name=name, sagemaker_session=sagemaker_session)
