# Databricks Setup — Sync S3 into a Volume

This document describes how to set up a Databricks Free Edition workspace and run
the S3 → Volume sync so that downstream Bronze ingestion can read raw parquet from
the Volume path rather than S3 directly.

## Why a Volume Instead of S3 Directly?

Databricks Free Edition serverless compute does not allow setting
`spark.hadoop.fs.s3a.access.key` at job/cluster start.  Spark cannot therefore
read `s3://` URIs natively.  The workaround is:

1. Run `sync_s3_to_volume` in a Python notebook using **boto3** (no Spark required).
2. Bronze ingestion reads from the Volume using standard `dbfs:/Volumes/…` paths.

## Prerequisites

- Databricks Free Edition account at <https://community.cloud.databricks.com/>.
- AWS credentials with `s3:GetObject`, `s3:ListBucket` access to `abhi-retail-demand-2026`.
- The raw S3 landing zone already populated (`make generate-data-s3-small` or full-scale).

---

## 1 — Create the Catalog, Schema, and Volume

In a Databricks SQL editor or notebook, run:

```sql
-- Use the pre-existing workspace catalog (available in Free Edition by default)
USE CATALOG workspace;

-- Create a dedicated schema for the project
CREATE SCHEMA IF NOT EXISTS retail_demand
  COMMENT 'Synthetic retail demand and inventory data';

-- Create the Volume that will hold raw parquet files synced from S3
CREATE VOLUME IF NOT EXISTS workspace.retail_demand.raw
  COMMENT 'Raw landing zone mirrored from S3';
```

The Volume is now accessible at `/Volumes/workspace/retail_demand/raw/`.

Verify:

```sql
DESCRIBE VOLUME workspace.retail_demand.raw;
```

---

## 2 — Install the Package in the Notebook

The sync notebook imports `retail_demand.io.databricks_sync`.  Install it at the
top of the notebook (or via the cluster library UI) with:

```python
%pip install git+https://github.com/<your-org>/retail-demand-inventory.git
```

Or copy `src/retail_demand/io/databricks_sync.py` directly into the notebook if
you prefer a self-contained script.

---

## 3 — Upload and Run the Sync Notebook

1. Open the Databricks workspace → **Workspace** → **Import**.
2. Select **File** → browse to
   `notebooks/databricks/00_sync_s3_to_volume.py.template`.
3. Import as **Python notebook**.
4. Open the notebook and **fill in the two PLACEHOLDER values** in Cell 2:
   ```python
   os.environ["AWS_ACCESS_KEY_ID"]     = "REPLACE_WITH_YOUR_ACCESS_KEY_ID"
   os.environ["AWS_SECRET_ACCESS_KEY"] = "REPLACE_WITH_YOUR_SECRET_ACCESS_KEY"
   ```
5. Run all cells (`Run all`).
6. Verify the summary table in the last cell shows all five tables synced.
7. **Delete the credential cell** (or replace it with `dbutils.secrets.get` calls)
   **before saving or sharing the notebook.**

> [!CAUTION]
> Never save AWS credentials in a Databricks notebook. Databricks workspaces are
> shared; secrets stored in plaintext cells are visible to any collaborator and
> can be captured in job run logs.

---

## 4 — Verify the Volume Contents

In a new cell or the Databricks SQL editor:

```python
display(dbutils.fs.ls("/Volumes/workspace/retail_demand/raw/"))
```

Expected output: five entries — `calendar/`, `prices/`, `products/`, `sales/`, `stores/`.

Spot-check row counts:

```python
from retail_demand.io.databricks_sync import sync_s3_to_volume  # already imported above

import pyarrow.dataset as ds

for table in ["stores", "products", "calendar", "prices", "sales"]:
    n = ds.dataset(f"/Volumes/workspace/retail_demand/raw/{table}", format="parquet",
                   partitioning="hive").count_rows()
    print(f"{table}: {n:,} rows")
```

---

## 5 — Re-running the Sync (Idempotency)

`sync_s3_to_volume` checks each destination file's size before downloading.
If the file already exists with the correct byte count it is skipped.
You can safely re-run the notebook after incremental S3 writes; only new or
changed objects are downloaded.

---

## 6 — Security Best Practice: Databricks Secrets

For production or shared workspaces, store credentials in a Databricks Secret Scope
instead of the notebook:

```bash
# From your local machine with the Databricks CLI configured:
databricks secrets create-scope retail-demand
databricks secrets put-secret retail-demand aws-access-key-id
databricks secrets put-secret retail-demand aws-secret-access-key
```

Then in the notebook Cell 2:

```python
AWS_ACCESS_KEY = dbutils.secrets.get("retail-demand", "aws-access-key-id")
AWS_SECRET_KEY = dbutils.secrets.get("retail-demand", "aws-secret-access-key")
```

Secrets are redacted in notebook output and job logs.
