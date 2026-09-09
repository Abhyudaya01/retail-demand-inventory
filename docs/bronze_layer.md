# Bronze layer

Bronze copies every raw row, including duplicates, nulls, negative values and other
intentional generator anomalies, into Delta. It performs no cleaning or deduplication.
Python logic lives in `src/retail_demand/bronze/`; the Databricks notebook only configures,
calls and displays results. Data quality rules belong in Phase 4.

## Schema fidelity

Explicit StructTypes match the generator's physical Parquet fields and reconstruct
Hive partition columns: sales uses year/month, prices uses year, dimensions are unpartitioned.
Strings, booleans, integer widths, doubles and stores.open_date (DATE) remain unchanged.

**Approved representation exception:** Phase 1 writes calendar.date, prices.week_start
and sales.date as Parquet `timestamp[ns]`. Spark 3.5/Delta cannot represent that logical
nanosecond timestamp type. Bronze reads their physical INT64 values as BIGINT epoch
nanoseconds, preserving every bit without rounding. These three fields are not DATE or
Spark TIMESTAMP columns. Any later conversion belongs in Silver. Locally the session
also enables `spark.sql.legacy.parquet.nanosAsLong`; reads always supply the explicit schema.

Explicit schemas avoid inference-dependent types; they are not a complete schema-drift
validator. Extra source fields can be ignored and absent fields can become null. Full
schema enforcement and business validation are outside this smoke-check scope.

## Metadata and write semantics

- `_ingested_at`: Spark current_timestamp at write time.
- `_source_path`: full source file URI, using input_file_name locally and the Databricks
  hidden `_metadata.file_path` column on Unity Catalog.
- `_ingest_run_id`: UUID4 hex string. An ingest_all call shares one ID across all five
  tables; each standalone ingest_table call gets a new ID.

Ingest order is stores, products, calendar, prices, sales. Overwrite replaces all rows,
including partitions absent from the next input. Append replays all raw rows; it is not
incremental ingestion or an idempotent retry. Statistics report rows and nonempty source
files written by this invocation, location, run ID and elapsed seconds. Files with zero
rows have no row lineage and are excluded from source_files.

Writes always use explicit Delta locations. Cloud-backed Bronze locations are also
registered as external Unity Catalog tables. Volume-backed Bronze locations skip
registration because Databricks Free Edition allows Delta files in Volumes but blocks
external table registration that points at Volume paths. That is fine for this project:
downstream Silver code reads Bronze by Delta path, using the `bronze_location` returned
by ingestion, so catalog registration is helpful metadata rather than a required contract.

When registration is enabled, create the schema first. Keep a table name associated with
one stable location: the requested CREATE TABLE IF NOT EXISTS statement does not relocate
an existing registration. Each table commits independently; ingest_all is not a transaction
across five tables. Do not run concurrent ingestions against the same destinations or
reuse one ingestor concurrently. A registration failure can leave the Delta files
successfully committed; inspect them before retrying, especially in append mode.

validate_bronze_table checks a minimum row count and present/non-null metadata, returning
row count, distinct run count, ISO timestamp bounds and checks_passed. It never repairs data.

## Local execution

Use an activated Python environment and Java 11 or 17. The first local Spark session may
need Maven access to download Delta JARs. The session uses local[2] and small shuffle counts;
extra_conf can override local configuration. Databricks uses its existing session unchanged.

```bash
source .venv/bin/activate
make install
make generate-data OUTPUT_PATH=data/synthetic N_STORES=2 N_SKUS=20 DAYS=60
make bronze-local
make lint
make test
```

bronze-local runs the actual pipeline, reading `data/synthetic` and writing `data/bronze`.
For other directories or append:

```bash
python -m retail_demand.bronze.cli --source-root data/synthetic_sample \
  --bronze-root data/bronze --mode append
```

Tests generate temporary input and output; they do not contact Databricks or real AWS.
The existing Phase 1/2 tests use their established AWS mocks.

## Databricks storage modes and next steps

Raw input remains `/Volumes/workspace/retail_demand/raw`.

Bronze supports two storage modes:

- Volume-backed, for Databricks Free Edition: set BRONZE_ROOT to
  `/Volumes/workspace/retail_demand/bronze`. Delta files are written under
  `/Volumes/workspace/retail_demand/bronze/{table_name}/`. Unity Catalog table registration
  is skipped because Free Edition blocks external table registration at Volume paths.
- Cloud-backed, for paid Databricks workspaces: set BRONZE_ROOT to a cloud storage prefix
  such as `s3://your-external-bucket/retail-demand-bronze`, covered by a Unity Catalog
  external location and outside all Volume and other table paths. Delta files are written
  by path and registered as external tables in `workspace.retail_demand_bronze`.

The ingestor rejects only unconfigured placeholders such as `REPLACE_WITH...`; Volume
destinations are valid. See [Volume path rules](https://docs.databricks.com/aws/en/volumes/paths).

1. Review and push the changes yourself. The notebook installs the GitHub version, so
   an unpushed local implementation is not available to it.
2. For Free Edition, confirm both raw and Bronze Volumes exist under
   `/Volumes/workspace/retail_demand/`. No external location is required because Bronze is
   read by path.
3. For a paid workspace using cloud-backed Bronze, confirm an external location is
   available. Have an administrator provision it if needed, with cloud storage authorization
   handled by Unity Catalog. Ensure USE CATALOG, USE SCHEMA, CREATE TABLE,
   CREATE EXTERNAL TABLE, and the appropriate READ FILES/WRITE FILES permissions. No AWS
   credentials belong in this notebook.
4. Confirm Phase 2 has synced all five raw datasets to the raw Volume.
5. Copy `notebooks/databricks/01_bronze_ingest.py.template` to a file named
   `01_bronze_ingest.py`, and import it as a Python notebook. COMMAND separators preserve
   notebook cells. The install and restart are separate executable cells so %pip is valid.
6. Run installation and restart. Keep BRONZE_ROOT as the Volume path for Free Edition, or
   replace it with the authorized cloud prefix for cloud-backed registration. Keep SOURCE_ROOT
   and SCHEMA_NAME as supplied. Use overwrite for the first run.
7. Run the remaining cells. Check per-table statistics, five PASS results and the sample
   rows. SHOW TABLES will show registered Bronze tables only in cloud-backed mode. The three
   documented date fields will display epoch nanoseconds.

The notebook has not been executed remotely. Local tests do not establish serverless
runtime compatibility. Databricks uses its bundled Spark/Delta runtime; verify package
installation compatibility there. The lineage expression follows the documented
[Unity Catalog replacement for input_file_name](https://docs.databricks.com/aws/en/data-governance/unity-catalog/jobs-update).
