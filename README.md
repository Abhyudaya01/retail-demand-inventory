# retail-demand-inventory

Retailers need enough inventory on the shelf to protect sales, but excess stock ties up cash,
fills back rooms, and increases markdown risk. This project models that tradeoff with a
portfolio-quality data platform: synthetic sales data is generated at store/SKU/day grain,
processed through analytics layers, forecasted, converted into reorder decisions, and surfaced
for operators in Power BI.

The synthetic data intentionally includes realistic business patterns and a small amount of
messiness. That gives downstream phases useful material for data quality checks, Spark
transformations, model validation, and executive-friendly inventory reporting without exposing
private retail data.

## Architecture Diagram

The current architecture placeholder is in [docs/architecture.md](docs/architecture.md).
AWS setup instructions are in [docs/aws_setup.md](docs/aws_setup.md).

## Stack

- AWS S3 for raw object storage
- Databricks for managed data engineering and ML workflows
- Delta Lake for Bronze, Silver, and Gold medallion tables
- PySpark for distributed transformations
- LightGBM for store/SKU demand forecasting
- MLflow for experiment tracking and model registry workflows
- Power BI for demand, inventory, and service-level dashboards

## Data Layout

The raw S3 landing zone is partitioned for efficient Spark and Databricks reads:

```text
s3://<bucket>/raw/
├── _manifests/
│   └── <run_id>.json
├── calendar/
│   └── part-00000-0.parquet
├── prices/
│   └── year=2026/
│       └── part-00000-0.parquet
├── products/
│   └── part-00000-0.parquet
├── sales/
│   └── year=2026/
│       └── month=1/
│           └── part-00000-0.parquet
└── stores/
    └── part-00000-0.parquet
```

Cost note: full-scale generation writes roughly 1 GB to S3, about $0.02/month in Standard
storage plus small one-time PUT request costs.

## Phase Checklist

- [x] Phase 1: Project scaffold and synthetic data generator
- [x] Phase 2: S3 raw landing zone and manifest
- [ ] Phase 3: AWS S3 landing-zone layout and ingestion contract
- [ ] Phase 4: Databricks Bronze Delta ingestion
- [ ] Phase 5: Silver cleaning, conformance, and enrichment
- [ ] Phase 6: Gold demand and inventory feature tables
- [ ] Phase 7: LightGBM forecasting baseline
- [ ] Phase 8: MLflow tracking and model evaluation
- [ ] Phase 9: Reorder-point and safety-stock decision logic
- [ ] Phase 10: Power BI semantic model and dashboard
- [ ] Phase 11: Portfolio polish, documentation, and deployment notes

## Reproducibility

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

Generate a small local sample:

```bash
python -m retail_demand.data_generation.generator \
  --target local \
  --output-path data/synthetic_sample \
  --n-stores 1 \
  --n-skus 10 \
  --days 90
```

Run linting and tests:

```bash
make lint
make test
```

For full-scale generation, omit the sizing flags. The defaults generate 50 stores, 2,000 SKUs,
and three years of daily sales, which produces more than 30 million sales rows and is intended
for the later Spark and Delta Lake phases.

Generate a small validation run against S3 after configuring `.env`:

```bash
make generate-data-s3-small
make verify-s3
```
