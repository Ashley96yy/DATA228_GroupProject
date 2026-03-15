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

## Notes

- The dataset is stored as partitioned Parquet.
- Do not download the full dataset to local disk unless you really need to.
- Prefer reading directly from `gs://data228/bts_ontime/` in cloud notebooks, Spark jobs, or other GCP environments.
