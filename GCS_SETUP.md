# GCS Dataset Access

This project stores the curated BTS on-time dataset in Google Cloud Storage.

## Project And Bucket

- GCP project ID: `digital-layout-475700-c2`
- GCS bucket: `gs://data228`
- Dataset root: `gs://data228/bts_ontime/`

## Access Requirements

You must be granted bucket access before these steps will work.

- Required role: `Storage Object Viewer`

## Install Google Cloud CLI

### macOS with Homebrew

```bash
brew install --cask google-cloud-sdk
```

Load `gcloud` into your shell:

```bash
source /opt/homebrew/share/google-cloud-sdk/path.zsh.inc
```

If you want this to load automatically in future terminals:

```bash
echo 'source /opt/homebrew/share/google-cloud-sdk/path.zsh.inc' >> ~/.zshrc
```

## Authenticate

Run:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project digital-layout-475700-c2
```

## Verify Access

Check that you can list the dataset:

```bash
gcloud storage ls gs://data228/bts_ontime/
```

If this works, your access is configured correctly.

## Read The Dataset

### PySpark

Read the full dataset:

```python
df = spark.read.parquet("gs://data228/bts_ontime/")
```

Read one year:

```python
df = spark.read.parquet("gs://data228/bts_ontime/year=2024")
```

Read one month:

```python
df = spark.read.parquet("gs://data228/bts_ontime/year=2024/month=01")
```

## Local Spark Directly Reading `gs://`

If you run Spark on your own laptop instead of Dataproc, you need the GCS connector jar.

### 1. Download the connector jar

Download a Hadoop 3 shaded GCS connector jar and save it as:

```text
third_party/gcs-connector-hadoop3-2.2.30-shaded.jar
```

### 2. Authenticate for local access

Run:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project digital-layout-475700-c2
```

### 3. Run Spark with the connector

Use the helper script in this repo:

```bash
./run_spark_with_gcs.sh test_read_gcs_parquet.py --path gs://data228/bts_ontime/year=2024/month=01/
```

The helper script automatically points Spark at:

```text
.gcloud-config/application_default_credentials.json
```

through `GOOGLE_APPLICATION_CREDENTIALS`, so it works with the repo-local ADC file created by `gcloud auth application-default login`.

Read the full dataset:

```bash
./run_spark_with_gcs.sh test_read_gcs_parquet.py --path gs://data228/bts_ontime/
```

### 4. If you are using Dataproc

You do not need this manual connector setup. Dataproc already installs the Cloud Storage connector.

## Notes

- The dataset is stored as partitioned Parquet.
- Do not download the full dataset to local disk unless you really need to.
- Prefer reading directly from `gs://data228/bts_ontime/` in cloud notebooks, Spark jobs, or other GCP environments.
