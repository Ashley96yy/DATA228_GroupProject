# DATA 228 Group Project

Flight delay risk prediction using the BTS on-time performance dataset.

This repo currently includes:

- data download and conversion scripts
- Spark feature engineering
- a baseline Spark ML model
- a FastAPI backend
- a lightweight frontend
- an MCP-style chatbox
- a minimal MCP stdio server

## Repo Layout

- `download_bts_ontime.py`: download raw BTS monthly ZIP files
- `convert_bts_ontime_to_parquet.py`: convert raw ZIP files to curated partitioned Parquet
- `build_bts_delay_features.py`: build the model-ready feature dataset
- `train_bts_delay_model.py`: train the baseline delay classifier
- `bts_delay_api.py`: FastAPI app for prediction and chat
- `bts_delay_mcp_server.py`: minimal stdio MCP server
- `frontend/index.html`: project UI
- `delay_tools.py`: shared prediction and route utility layer
- `GCS_SETUP.md`: GCS access instructions
- `MCP_SETUP.md`: MCP and chat setup instructions

## What Is In Git And What Is Not

This repo does not include large local artifacts.

Not committed:

- raw BTS data
- curated Parquet data
- feature data
- trained model files
- `.venv`

That means a fresh clone will not be enough by itself to run the full app. You also need either:

1. a shared trained model directory, or
2. access to the dataset in GCS and the ability to rebuild the feature/model pipeline locally

Current shared GCS artifact paths:

- curated dataset: `gs://data228/bts_ontime/`
- feature dataset: `gs://data228/feature/bts_delay_15/`
- trained model: `gs://data228/model/bts_delay_lr_baseline/`

## Quick Start For Teammates

Use this path if someone mainly wants to run the UI and API, not rebuild the entire pipeline.

### 1. Clone the repo

```bash
git clone git@github.com:Ashley96yy/DATA228_GroupProject.git
cd DATA228_GroupProject
```

### 2. Create a local virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn pyspark==3.5.8 setuptools
```

### 3. Get the trained model directory

You need a local copy of:

```text
data/model/bts_delay_lr_baseline
```

If this folder is missing, the API cannot start.

Shared model location:

```text
gs://data228/model/bts_delay_lr_baseline/
```

Example copy command:

```bash
gcloud storage cp -r gs://data228/model/bts_delay_lr_baseline data/model/
```

### 4. Start the API and UI

```bash
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

Open:

```text
http://127.0.0.1:8000/app
```

### 5. Optional: enable LLM chat

If you want the chatbox to use OpenAI for tool selection:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5-mini
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

If `OPENAI_API_KEY` is not set, the chat endpoint still works, but it falls back to a local rule-based router.

## Full Reproducible Pipeline

Use this path if you want to rebuild the project artifacts from data.

### 1. Get GCS access

Follow:

- [GCS_SETUP.md](/Users/dingyuyao/Documents/SJSU/Spring2026/DATA228/GroupProject/GCS_SETUP.md)

Dataset root:

```text
gs://data228/bts_ontime/
```

Shared feature dataset:

```text
gs://data228/feature/bts_delay_15/
```

Shared trained model:

```text
gs://data228/model/bts_delay_lr_baseline/
```

### 2. Read the curated dataset from GCS

If you are running Spark locally against `gs://`, also follow the local connector setup in:

- [GCS_SETUP.md](/Users/dingyuyao/Documents/SJSU/Spring2026/DATA228/GroupProject/GCS_SETUP.md)

### 3. Build the feature dataset

Local curated data path example:

```bash
spark-submit build_bts_delay_features.py \
  --input data/curated/bts_ontime \
  --output data/feature/bts_delay_15 \
  --overwrite
```

Directly from GCS:

```bash
./run_spark_with_gcs.sh build_bts_delay_features.py \
  --input gs://data228/bts_ontime \
  --output data/feature/bts_delay_15 \
  --overwrite
```

If you only need the already-generated feature layer instead of rebuilding it:

```bash
gcloud storage cp -r gs://data228/feature/bts_delay_15 data/feature/
```

### 4. Train the baseline model

Example baseline run that is small enough for a laptop:

```bash
spark-submit train_bts_delay_model.py \
  --input data/feature/bts_delay_15 \
  --output-dir data/model/bts_delay_lr_baseline \
  --train-sample-frac 0.002 \
  --valid-sample-frac 0.01 \
  --test-sample-frac 0.01 \
  --overwrite
```

If you only need the already-trained model instead of retraining it:

```bash
gcloud storage cp -r gs://data228/model/bts_delay_lr_baseline data/model/
```

### 5. Start the API and UI

```bash
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

Open:

```text
http://127.0.0.1:8000/app
```

## MCP And Chat

This project includes both:

- a browser chatbox exposed through `/chat`
- a minimal stdio MCP server

See:

- [MCP_SETUP.md](/Users/dingyuyao/Documents/SJSU/Spring2026/DATA228/GroupProject/MCP_SETUP.md)

Run the MCP server:

```bash
source .venv/bin/activate
python bts_delay_mcp_server.py
```

## Current Model Scope

The current baseline model predicts whether a flight will arrive at least 15 minutes late.

It uses scheduled-flight information available before departure, including:

- date context
- scheduled departure and arrival time
- airline
- route
- derived route distance
- derived elapsed time

Training coverage:

- historical BTS data from `1987` through `2025`

This means future scheduled flights can still be scored using learned historical patterns.

## Notes

- This is a baseline model, not a real-time operational delay system.
- The UI is designed for demoability first.
- The current repo is runnable with the right local model files, but true end-to-end reproducibility still depends on data and model access being shared clearly across the team.
