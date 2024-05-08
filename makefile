.PHONY: quality style unit-test integ-test

check_dirs := src tests

# run tests

unit-test:
	python -m pytest -v -s  ./tests/unit/

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