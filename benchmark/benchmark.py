import time
import boto3
import os
import json
from locust import User, task, between

# from locust import task, between

os.environ["AWS_PROFILE"] = "hf-sm"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

client = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", None)

if ENDPOINT_NAME is None:
    raise EnvironmentError("you have to define ENV ENDPOINT_NAME to run your benchmark")

# INPUT = {
#     "inputs": "Jeff: Can I train a Hugging Face Transformers model on Amazon SageMaker? Philipp: Sure you can use the new Hugging Face Deep Learning Container. Jeff: ok. Jeff: and how can I get started? Jeff: where can I find documentation?  Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face. What a nice blog"
# }
INPUT = {
    "inputs": {
        "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for infernece.",
        "question": "Where lives Philipp?",
    }
}


class SageMakerClient:
    _locust_environment = None

    def __init__(self):
        super().__init__()
        self.client = boto3.client("sagemaker-runtime")

    @staticmethod
    def total_time(start_time) -> float:
        return int((time.time() - start_time) * 1000)

    def send(self, name: str, payload: dict, endpoint_name: str):
        start_time = time.time()
        try:
            response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(INPUT),
                ContentType="application/json",
                Accept="application/json",
            )
        except Exception as e:
            self._locust_environment.request_failure.fire(
                request_type="execute", name=name, response_time=self.total_time(start_time), exception=e
            )
        self._locust_environment.events.request_success.fire(
            request_type="execute", name=name, response_time=self.total_time(start_time), response_length=0
        )


class SageMakerUser(User):
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = SageMakerClient()
        self.client._locust_environment = self.environment


class SimpleSendRequest(SageMakerUser):
    wait_time = between(0.1, 1)

    @task
    def send_request(self):
        payload = INPUT
        self.client.send("send_endpoint", payload, ENDPOINT_NAME)
