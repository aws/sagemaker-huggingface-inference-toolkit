.PHONY: build test push quality style unit-test integ-test


# runtime=starlette
runtime=cpu
framework=pytorch
container_name=huggingface-inference-${framework}
aws_region=us-east-1
aws_account_id=558105141721
aws_profile=hf-sm
ecr_url=${aws_account_id}.dkr.ecr.${aws_region}.amazonaws.com
tag=${runtime}-0.0.1
cwd  := $(shell pwd)

# build local container

build:
	docker build --tag ${ecr_url}/${container_name}:${tag} \
							 --build-arg TRANSFORMERS_VERSION=4.6.1 \
							 --file ./docker/${framework}/1.7/py3//Dockerfile.${runtime} \
							 .

# pushes local container to aws ecr 

push:
	#aws ecr create-repository --repository-name ${container_name}  --profile ${aws_profile} --region ${aws_region}   > /dev/null
	aws ecr get-login-password \
        --region ${aws_region} \
        --profile ${aws_profile} \
    | docker login \
        --username AWS \
        --password-stdin ${ecr_url}
	docker push ${ecr_url}/${container_name}:${tag} 

# build and deploys container

deploy: build push

# Starts local container

start:
	# docker run -t -i --env HF_TASK="text-classification" -p 8080:8080 -v ${cwd}/model:/opt/ml/model ${ecr_url}/${container_name}:${tag} 
	# docker run -t -i --env HF_TASK="text-classification" --env HF_MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english" -p 8080:8080  ${ecr_url}/${container_name}:${tag} 
	docker run -t -i --env HF_TASK="question-answering" --env HF_MODEL_ID="sshleifer/tiny-distilbert-base-cased-distilled-squad" -p 8080:8080  ${ecr_url}/${container_name}:${tag} 
	# docker run -t -i --env HF_TASK="summarization" --env HF_MODEL_ID="philschmid/distilbart-cnn-12-6-samsum" -p 8080:8080  ${ecr_url}/${container_name}:${tag} 
	# docker run -t -i --env HF_TASK="zero-shot-classification" --env HFMODEL_ID="joeddav/xlm-roberta-large-xnli" -p 8080:8080  ${ecr_url}/${container_name}:${tag} 
	# docker run -t -i --env HF_TASK="ner" --env HF_MODEL_ID="dbmdz/bert-large-cased-finetuned-conll03-english" -p 8080:8080  ${ecr_url}/${container_name}:${tag} 
	# docker run -t -i --env HF_TASK="translation_xx_to_yy" --env HF_MODEL_ID="Helsinki-NLP/opus-mt-en-de" -p 8080:8080  ${ecr_url}/${container_name}:${tag} 


bash:
	docker run -t -i -it -v ${cwd}/model:/opt/ml/model ${ecr_url}/${container_name}:${tag} /bin/bash 


check_dirs := src deployment docker tests

# run tests


unit-test:
	python -m pytest -n auto --dist loadfile -s -v ./tests/unit/

integ-test:
	python -m pytest -n 2 -s -v ./tests/integ/
	# python -m pytest -n auto -s -v ./tests/integ/


# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py36 $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# Format source code automatically

style:
	# black --line-length 119 --target-version py36 tests src benchmarks datasets metrics
	black --line-length 119 --target-version py36 $(check_dirs)
	isort $(check_dirs)