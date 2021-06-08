import argparse
import os
from uuid import uuid4

from sagemaker import Session
from sagemaker.model import Model


ROLE_NAME = "sagemaker_execution_role"
INITIAL_INSTANCE_COUNT = 1
PROFILE = "hf-sm"
REGION = "us-east-1"


def deploy(image_uri, model_data, role_name, env, deploy_config, profile, region, name):

    os.environ["AWS_PROFILE"] = profile  # setting aws profile for our boto3 client
    os.environ["AWS_DEFAULT_REGION"] = region  # current DLCs are only in us-east-1 available
    sess = Session()

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


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--image_uri", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_data", type=str, default=None)
    parser.add_argument("--hf_task", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--instance_type", type=str
    )  # "instance_type": "ml.m5.large",  # ‘ml.p2.xlarge’, 'ml.g4dn.xlarge' ,'ml.inf1.xlarge', or ‘local’ for local mode.
    parser.add_argument("--initial_instance_count", type=int, default=INITIAL_INSTANCE_COUNT)
    parser.add_argument("--profile", type=str, default=PROFILE)
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--role_name", type=str, default=ROLE_NAME)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    env = {}
    if args.hf_task is not None:
        env["HF_TASK"] = args.hf_task

    if args.model_id is not None:
        env["HF_MODEL_ID"] = args.model_id
    if args.model_data is None and args.model_id is None:
        raise ValueError("You need to provide either `model_id` or `model_data`")

    if args.name is not None:
        name = f"{args.name}-{str(uuid4())}"
    elif args.model_id is not None:
        name = f"{args.model_id.replace('/','-')}-{str(uuid4())}"
    else:
        name = f"huggingface-inference-{str(uuid4())}"

    name = name[:62] if len(name) > 62 else name

    deploy_config = {"instance_type": args.instance_type, "initial_instance_count": args.initial_instance_count}

    deploy(args.image_uri, args.model_data, args.role_name, env, deploy_config, args.profile, args.region, name)
