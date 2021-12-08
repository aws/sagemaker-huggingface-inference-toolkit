import json
import os
import re

import numpy as np
import pytest

import boto3
from integ.config import task2input, task2model, task2output, task2performance, task2validation
from integ.utils import clean_up, count_tokens, timeout_and_delete_by_name, track_infer_time
from sagemaker import Session
from sagemaker.model import Model


os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
SAGEMAKER_EXECUTION_ROLE = os.environ.get("SAGEMAKER_EXECUTION_ROLE", "sagemaker_execution_role")


def get_framework_ecr_image(registry_id="763104351884", repository_name="huggingface-pytorch-inference", device="cpu"):
    client = boto3.client("ecr")

    def get_all_ecr_images(registry_id, repository_name, result_key):
        response = client.list_images(
            registryId=registry_id,
            repositoryName=repository_name,
        )
        results = response[result_key]
        while "nextToken" in response:
            response = client.list_images(
                registryId=registry_id,
                nextToken=response["nextToken"],
                repositoryName=repository_name,
            )
            results.extend(response[result_key])
        return results

    images = get_all_ecr_images(registry_id=registry_id, repository_name=repository_name, result_key="imageIds")
    image_tags = [image["imageTag"] for image in images]
    print(image_tags)
    image_regex = re.compile("\d\.\d\.\d-" + device + "-.{4}$")
    tag = sorted(list(filter(image_regex.match, image_tags)), reverse=True)[0]
    return f"{registry_id}.dkr.ecr.{os.environ.get('AWS_DEFAULT_REGION','us-east-1')}.amazonaws.com/{repository_name}:{tag}"


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
    ["pytorch", "tensorflow"],
)
@pytest.mark.parametrize(
    "device",
    [
        "gpu",
        "cpu",
    ],
)
def test_deployment_from_hub(task, device, framework):
    image_uri = get_framework_ecr_image(repository_name=f"huggingface-{framework}-inference", device=device)
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
        role=SAGEMAKER_EXECUTION_ROLE,  # An AWS IAM role (either name or full ARN).
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
            print(data)
            json.dump(data, outfile)

        assert task2performance[task][device]["average_request_time"] >= np.mean(time_buffer)

        clean_up(endpoint_name=name, sagemaker_session=sagemaker_session)
