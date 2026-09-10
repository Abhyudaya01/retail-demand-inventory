PYTHON ?= python
OUTPUT_PATH ?= data/synthetic_sample
N_STORES ?= 1
N_SKUS ?= 10
DAYS ?= 90
S3_MODE ?= overwrite
SPARK_IVY_DIR ?= $(CURDIR)/.spark-ivy
PYTEST_ARGS ?=
ERROR_ANALYSIS_FIXTURE_DIR ?= tests/fixtures/error_analysis
ERROR_ANALYSIS_KERNEL_DIR ?= reports/.jupyter/kernels/phase8
ERROR_ANALYSIS_PYTHON ?= $(shell $(PYTHON) -c "import sys; print(sys.executable)")
INVENTORY_FIXTURE_DIR ?= tests/fixtures/error_analysis
INVENTORY_KERNEL_DIR ?= reports/.jupyter/kernels/phase9
INVENTORY_PYTHON ?= $(shell $(PYTHON) -c "import sys; print(sys.executable)")

.PHONY: install lint test generate-data error-analysis-local inventory-local

install:
	$(PYTHON) -m pip install -e ".[local-spark]"

lint:
	$(PYTHON) -m ruff check .

test:
	mkdir -p $(SPARK_IVY_DIR)
	PYSPARK_SUBMIT_ARGS="--conf spark.jars.ivy=$(SPARK_IVY_DIR) pyspark-shell" $(PYTHON) -m pytest $(PYTEST_ARGS)

error-analysis-local:
	mkdir -p reports/figures reports/tables
	mkdir -p $(ERROR_ANALYSIS_KERNEL_DIR)
	printf '%s\n' \
		'{' \
		'  "argv": ["$(ERROR_ANALYSIS_PYTHON)", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
		'  "display_name": "Phase 8 Error Analysis", "language": "python"' \
		'}' > $(ERROR_ANALYSIS_KERNEL_DIR)/kernel.json
	ERROR_ANALYSIS_PREDICTIONS_CSV=$(ERROR_ANALYSIS_FIXTURE_DIR)/forecast_vs_actual.csv \
	ERROR_ANALYSIS_SALES_CSV=$(ERROR_ANALYSIS_FIXTURE_DIR)/sales_history.csv \
	ERROR_ANALYSIS_PRICES_CSV=$(ERROR_ANALYSIS_FIXTURE_DIR)/prices.csv \
	ERROR_ANALYSIS_STORES_CSV=$(ERROR_ANALYSIS_FIXTURE_DIR)/stores.csv \
	ERROR_ANALYSIS_PRODUCTS_CSV=$(ERROR_ANALYSIS_FIXTURE_DIR)/products.csv \
	ERROR_ANALYSIS_OUTPUT_DIR=reports/figures \
	ERROR_ANALYSIS_TABLE_DIR=reports/tables \
	JUPYTER_PATH=$(CURDIR)/reports/.jupyter \
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/analysis/06_error_analysis.ipynb \
		--ExecutePreprocessor.kernel_name=phase8 \
		--output 06_error_analysis.executed.ipynb \
		--output-dir reports

inventory-local:
	mkdir -p reports/figures reports/tables
	rm -f reports/tables/inventory_*.csv reports/figures/inventory_*.png
	mkdir -p $(INVENTORY_KERNEL_DIR)
	printf '%s\n' \
		'{' \
		'  "argv": ["$(INVENTORY_PYTHON)", "-m", "ipykernel_launcher", "-f", "{connection_file}"],' \
		'  "display_name": "Phase 9 Inventory Decision", "language": "python"' \
		'}' > $(INVENTORY_KERNEL_DIR)/kernel.json
	INVENTORY_PREDICTIONS_CSV=$(abspath $(INVENTORY_FIXTURE_DIR)/forecast_vs_actual.csv) \
	INVENTORY_SALES_CSV=$(abspath $(INVENTORY_FIXTURE_DIR)/sales_history.csv) \
	INVENTORY_PRICES_CSV=$(abspath $(INVENTORY_FIXTURE_DIR)/prices.csv) \
	INVENTORY_PRODUCTS_CSV=$(abspath $(INVENTORY_FIXTURE_DIR)/products.csv) \
	INVENTORY_FIGURE_DIR=$(abspath reports/figures) \
	INVENTORY_TABLE_DIR=$(abspath reports/tables) \
	JUPYTER_PATH=$(CURDIR)/reports/.jupyter \
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/analysis/07_inventory_decision.ipynb \
		--ExecutePreprocessor.kernel_name=phase9 \
		--output 07_inventory_decision.executed.ipynb \
		--output-dir reports

generate-data:
	$(PYTHON) -m retail_demand.data_generation.generator --target local --output-path $(OUTPUT_PATH) --n-stores $(N_STORES) --n-skus $(N_SKUS) --days $(DAYS)

.PHONY: generate-data-s3 generate-data-s3-small verify-s3

generate-data-s3:
	$(PYTHON) -m retail_demand.data_generation.generator --target s3 --mode $(S3_MODE)

generate-data-s3-small:
	$(PYTHON) -m retail_demand.data_generation.generator --target s3 --mode $(S3_MODE) --n-stores 5 --n-skus 200 --days 365

verify-s3:
	$(PYTHON) -m retail_demand.io.s3_writer verify

.PHONY: bronze-local silver-local gold-local baselines-local gbm-local
bronze-local:
	$(PYTHON) -m retail_demand.bronze.cli

silver-local:
	$(PYTHON) -m retail_demand.silver.cli

gold-local:
	$(PYTHON) -m retail_demand.gold.cli

baselines-local:
	MLFLOW_TRACKING_URI=file:///tmp/retail-demand-mlruns $(PYTHON) -m retail_demand.experiments.run_baselines \
		--gold-root data/gold \
		--experiment-path retail-demand-baselines-local


gbm-local:
	MLFLOW_TRACKING_URI=file:///tmp/retail-demand-mlruns $(PYTHON) -m retail_demand.experiments.run_gbm \
		--gold-root data/gold \
		--experiment-path retail-demand-gbm-local \
		--cv-folds 1 \
		--horizon-days 14 \
		--min-train-days 180 \
		--sample-rows 5000
