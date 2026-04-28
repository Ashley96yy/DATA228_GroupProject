# DATA 228 Group Project

Flight delay risk prediction using the BTS on-time performance dataset.

This repo currently includes:

- data download and conversion scripts
- Spark feature engineering
- baseline and multi-model Spark ML training workflows
- a FastAPI backend
- a single-model frontend and a multi-model frontend
- an MCP-style chatbox
- a minimal MCP stdio server

## Repo Layout

- `download_bts_ontime.py`: download raw BTS monthly ZIP files
- `convert_bts_ontime_to_parquet.py`: convert raw ZIP files to curated partitioned Parquet
- `build_bts_delay_features.py`: build the model-ready feature dataset
- `train_bts_delay_model.py`: train the baseline delay classifier
- `train_bts_delay_mlflow.py`: compare multiple models with MLflow tracking
- `bts_delay_api.py`: FastAPI app for prediction and chat
- `bts_delay_mcp_server.py`: minimal stdio MCP server
- `frontend/index.html`: original single-model UI
- `frontend/index_multimodel.html`: multi-model UI with deployed-model selection
- `delay_tools.py`: shared prediction and route utility layer
- `GCS_SETUP.md`: GCS access instructions
- `MCP_SETUP.md`: MCP and chat setup instructions
- `MLFLOW_PIPELINE_GUIDE.md`: MLflow training, comparison, and deployment guide

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

Additional local model directories currently used by the multi-model UI:

- `data/model/bts_delay_lr_baseline`
- `data/model/bts_delay_rf_best`
- `data/model/bts_delay_gbt_best`
- `data/model/bts_delay_best_recent_3models`

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

### 3. Get the required model directories

At minimum, the original single-model UI needs:

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

If you want the multi-model UI to work, you also need these local directories:

```text
data/model/bts_delay_rf_best
data/model/bts_delay_gbt_best
data/model/bts_delay_best_recent_3models
```

The multi-model page uses:

- `lr`, `rf`, and `gbt` as switchable deployed models
- `data/model/bts_delay_best_recent_3models` to infer the current recommended model

### 4. Start the API and UI

```bash
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

Open either UI:

```text
http://127.0.0.1:8000/app
http://127.0.0.1:8000/app_multimodel
```

Use:

- `/app` for the original single-model baseline page
- `/app_multimodel` for the deployed multi-model selector page

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

### 5. Train and export multi-model artifacts

For the multi-model workflow, the currently expected local directories are:

```text
data/model/bts_delay_rf_best
data/model/bts_delay_gbt_best
data/model/bts_delay_best_recent_3models
```

See:

- [MLFLOW_PIPELINE_GUIDE.md](/Users/dingyuyao/Documents/SJSU/Spring2026/DATA228/GroupProject/MLFLOW_PIPELINE_GUIDE.md)

### 6. Start the API and UI

```bash
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

Open either UI:

```text
http://127.0.0.1:8000/app
http://127.0.0.1:8000/app_multimodel
```

## MCP And Chat

This project includes both:

- a browser chatbox exposed through `/chat`
- a minimal stdio MCP server

The chat endpoint now follows the selected model from the multi-model form. If the page is using `rf`, the chat requests use `rf` too.

See:

- [MCP_SETUP.md](/Users/dingyuyao/Documents/SJSU/Spring2026/DATA228/GroupProject/MCP_SETUP.md)

Run the MCP server:

```bash
source .venv/bin/activate
python bts_delay_mcp_server.py
```

## Current Model Scope

The project predicts whether a flight will arrive at least 15 minutes late.

Available deployed model families:

- Logistic Regression
- Random Forest
- Gradient-Boosted Trees

The deployed models use scheduled-flight information available before departure, including:

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

- This is a historical-pattern delay prediction system, not a real-time operational delay system.
- `/app` is the stable single-model baseline page.
- `/app_multimodel` is the multi-model deployment page.
- The multi-model page shows a `recommended model` derived from the exported best-model directory, not from a hardcoded UI constant.
- The current repo is runnable with the right local model files, but true end-to-end reproducibility still depends on data and model access being shared clearly across the team.
