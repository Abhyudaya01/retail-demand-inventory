# Silver layer

Silver turns path-backed Bronze Delta files into clean, typed Delta tables. It keeps the
same five-table shape as Bronze, writes to `/Volumes/workspace/retail_demand/silver/{table}`,
and reads all inputs by Delta path. No Bronze or Silver table is required to be registered in
Unity Catalog for Phase 4.

## Cleaning rules

- stores, products and calendar: cast to the strict Silver schema, drop rows with nulls in
  required columns, and deduplicate natural keys.
- prices: lowercase store IDs, cast to the strict schema, fill null `price` values from the
  prior non-null price for the same `(store_id, sku_id)` ordered by `week_start`, drop rows
  where no prior price exists, and deduplicate `(store_id, sku_id, week_start)`.
- sales: lowercase store IDs, cast to the strict schema, drop rows with null required fields,
  drop negative `units_sold`, and deduplicate `(store_id, sku_id, date)` by keeping the row
  with the earliest Bronze `_ingested_at`.

Each cleaning function returns a cleaned Spark DataFrame and metrics:
`rows_in`, `rows_out`, `dupes_dropped`, `nulls_dropped`, `invalid_dropped`, and
`imputations_applied`. The builder logs those metrics at INFO and adds `_silvered_at`,
`_silver_run_id`, and `_bronze_run_id` to each output row.

The Bronze nanosecond timestamp fields remain BIGINT epoch nanoseconds in Silver:
`calendar.date`, `prices.week_start`, and `sales.date`. Calendar conversion to a business
date can happen in Gold or reporting-facing tables once the desired type contract is fixed.

## Data quality suite

`SilverDQ.run_all_checks(spark, silver_root)` loads the five Silver Delta paths and runs:

- no nulls on every natural key.
- uniqueness on every natural key.
- sales.store_id in stores.store_id.
- sales.sku_id in products.sku_id.
- sales.date in calendar.date.
- prices.store_id in stores.store_id.
- prices.sku_id in products.sku_id.

Each check returns `check_name`, `passed`, `details`, and `timestamp`. Any `passed = False`
means the Silver layer is not trustworthy for downstream feature engineering. Referential
integrity failures include `orphan_count`; uniqueness and null failures include the key
columns in `details`.

## Local execution

Start with generated raw data and a Bronze build:

```bash
source .venv/bin/activate
make generate-data OUTPUT_PATH=data/synthetic N_STORES=2 N_SKUS=20 DAYS=60
make bronze-local
make silver-local
make test
```

`make silver-local` reads `data/bronze`, writes `data/silver`, and runs the full DQ suite.
For custom paths:

```bash
python -m retail_demand.silver.cli --bronze-root data/bronze --silver-root data/silver
```

## Databricks execution

The Databricks notebook installs the project with `--no-deps`, then installs only the
non-Spark Python dependencies. Databricks serverless provides `pyspark` and Delta support.
The notebook reads Bronze from `/Volumes/workspace/retail_demand/bronze` and writes Silver to
`/Volumes/workspace/retail_demand/silver`, then renders the DQ results and raises if any
critical check fails.

