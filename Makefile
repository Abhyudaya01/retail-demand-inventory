PYTHON ?= python
OUTPUT_PATH ?= data/synthetic_sample
N_STORES ?= 1
N_SKUS ?= 10
DAYS ?= 90
S3_MODE ?= overwrite

.PHONY: install lint test generate-data

install:
	$(PYTHON) -m pip install -e ".[local-spark]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

generate-data:
	$(PYTHON) -m retail_demand.data_generation.generator --target local --output-path $(OUTPUT_PATH) --n-stores $(N_STORES) --n-skus $(N_SKUS) --days $(DAYS)

.PHONY: generate-data-s3 generate-data-s3-small verify-s3

generate-data-s3:
	$(PYTHON) -m retail_demand.data_generation.generator --target s3 --mode $(S3_MODE)

generate-data-s3-small:
	$(PYTHON) -m retail_demand.data_generation.generator --target s3 --mode $(S3_MODE) --n-stores 5 --n-skus 200 --days 365

verify-s3:
	$(PYTHON) -m retail_demand.io.s3_writer verify

.PHONY: bronze-local
bronze-local:
	$(PYTHON) -m retail_demand.bronze.cli
