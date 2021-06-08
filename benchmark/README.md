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

[Locust](https://docs.locust.io/en/stable/what-is-locust.html) is an easy to use, scriptable and scalable performance testing tool.
You define the behaviour of your users in regular Python code, instead of using a clunky UI or domain specific language.
This makes Locust infinitely expandable and very developer friendly.

## Setup

```bash
pip install ".[benchmark]"
```

## Run 

### with UI 

Since we are using a custom client for the request we need to define the "Host" as `-`.


```bash
ENDPOINT_NAME="distilbert-base-uncased-distilled-squad-6493832c-767d-4cdb-a9a" locust -f benchmark.py 
```

### headless

```bash
--users  Number of concurrent Locust users
--spawn-rate  The rate per second in which users are spawned until num users
--run-time duration of test
```


```bash 
ENDPOINT_NAME="distilbert-base-uncased-distilled-squad-6493832c-767d-4cdb-a9a" locust -f benchmark.py \
       --users 6 \
       --spawn-rate 1 \
       --run-time 180s \
       --headless
```