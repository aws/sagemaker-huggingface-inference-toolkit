import json
import os
import re

import numpy as np
import pytest

import boto3
from integ.utils import clean_up, timeout_and_delete_by_name
from sagemaker import Session
from sagemaker.model import Model
from io import BytesIO
from PIL import Image


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


# TODO: needs existing container
def test_text_to_image_model():
    image_uri = get_framework_ecr_image(repository_name=f"huggingface-pytorch-inference", device="gpu")

    name = f"hf-test-text-to-image"
    task = "text-to-image"
    model = "echarlaix/tiny-random-stable-diffusion-xl"
    # instance_type = "ml.m5.large" if device == "cpu" else "ml.g4dn.xlarge"
    instance_type = "local_gpu"
    number_of_requests = 100
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
        response = client.invoke_endpoint(
            EndpointName=name,
            Body={"inputs": "a yellow lemon tree"},
            ContentType="application/json",
            Accept="image/png",
        )

        # validate response
        response_body = response["Body"].read().decode("utf-8")

        img = Image.open(BytesIO(response_body))
        assert isinstance(img, Image.Image)

        clean_up(endpoint_name=name, sagemaker_session=sagemaker_session)
