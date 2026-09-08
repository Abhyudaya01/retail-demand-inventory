# AWS Setup

This document records every step taken to create the AWS resources used by the
retail-demand project.  Follow it from scratch to reproduce the exact environment.

---

## 1 — Create an IAM User

1. Sign in to the [AWS Console](https://console.aws.amazon.com/) and navigate to
   **IAM → Users → Create user**.
2. User name: `retail-demand-cli`.
3. Skip console access; this user is CLI-only.
4. On the **Set permissions** screen choose **Attach policies directly** and attach
   **AmazonS3FullAccess**.  (You can tighten to a least-privilege policy later.)
5. Click **Create user**.

Create an access key for the user:

1. Open the newly created user → **Security credentials** → **Create access key**.
2. Use case: **Command Line Interface (CLI)**.
3. Download the CSV or copy **Access key ID** and **Secret access key** immediately —
   the secret is not shown again.

---

## 2 — Create the S3 Bucket

Create one dedicated bucket for this project in `us-east-1`:

```bash
aws s3api create-bucket \
  --bucket abhi-retail-demand-2026 \
  --region us-east-1
```

For regions other than `us-east-1`, include the location constraint:

```bash
aws s3api create-bucket \
  --bucket <bucket> \
  --region us-west-2 \
  --create-bucket-configuration LocationConstraint=us-west-2
```

Enable **Block all public access** (it is on by default for new buckets, but verify):

```bash
aws s3api put-public-access-block \
  --bucket abhi-retail-demand-2026 \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## 3 — Minimal IAM Policy (Optional Hardening)

Replace AmazonS3FullAccess with a scoped policy that allows only the actions the
generator needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListRetailDemandBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::abhi-retail-demand-2026"
    },
    {
      "Sid": "ReadWriteRetailDemandObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::abhi-retail-demand-2026/*"
    }
  ]
}
```

The overwrite safety check in `S3ParquetWriter` prevents accidental deletion of
Bronze, Silver, or Gold prefixes.

---

## 4 — Install the AWS CLI

```bash
# macOS (Homebrew)
brew install awscli

# Or with pip
pip install awscli
```

Verify the installation:

```bash
aws --version
```

---

## 5 — Configure the Named Profile

```bash
aws configure --profile retail-demand
```

When prompted, enter:

| Prompt | Value |
|---|---|
| AWS Access Key ID | *the key from step 1* |
| AWS Secret Access Key | *the secret from step 1* |
| Default region name | `us-east-1` |
| Default output format | `json` |

This writes credentials to `~/.aws/credentials` and config to `~/.aws/config`.

---

## 6 — Create the Project `.env` File

Copy `.env.example` to `.env` and fill in your bucket name:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# AWS_PROFILE takes precedence over inline AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
AWS_PROFILE=retail-demand
AWS_REGION=us-east-1
S3_BUCKET=abhi-retail-demand-2026
S3_RAW_PREFIX=raw
S3_BRONZE_PREFIX=bronze
S3_SILVER_PREFIX=silver
S3_GOLD_PREFIX=gold
DATABRICKS_VOLUME_ROOT=/Volumes/workspace/retail_demand/raw
RANDOM_SEED=42
```

---

## 7 — Verification

Confirm the profile can reach the bucket:

```bash
aws s3 ls --profile retail-demand
# Should list: abhi-retail-demand-2026
```

Generate a small test dataset to S3 and verify:

```bash
make generate-data-s3-small
make verify-s3
```

After a full generation run, browse the raw landing zone:

```bash
aws s3 ls s3://abhi-retail-demand-2026/raw/ --recursive --profile retail-demand
```

Expected layout:

```text
s3://abhi-retail-demand-2026/raw/
├── _manifests/<run_id>.json
├── calendar/part-00000-0.parquet
├── prices/year=.../part-*.parquet
├── products/part-00000-0.parquet
├── sales/year=.../month=.../part-*.parquet
└── stores/part-00000-0.parquet
```
