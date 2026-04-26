# MLFLOW PIPELINE GUIDE

This project includes an MLflow-based training workflow so the team can show:

- reproducible experiment tracking
- comparison across multiple models instead of only one baseline
- deployment of the selected best model into the FastAPI app

## Key Paths

This guide does not assume a specific folder layout. The important paths are:

- `PROJECT_ROOT`: the repository root that contains `train_bts_delay_mlflow.py`
- `PROJECT_ROOT/.venv/bin/python`: the Python interpreter used for Spark and MLflow
- `PROJECT_ROOT/run_spark_with_gcs.sh`: helper script for Spark jobs that read `gs://` paths
- `PROJECT_ROOT/data/feature/bts_delay_15/`: local copy of the prepared feature dataset
- `PROJECT_ROOT/data/model/`: exported trained models
- `PROJECT_ROOT/mlruns/`: local MLflow tracking store when using `file:./mlruns`
- `GOOGLE_APPLICATION_CREDENTIALS`: Application Default Credentials JSON for Google Cloud
- `CONNECTOR_JAR`: the GCS Spark connector jar, typically `gcs-connector-hadoop3-2.2.30-shaded.jar`

## Files Added for Reproducibility

- `train_bts_delay_mlflow.py`: tracked MLflow experiment runner
- `MLproject`: reproducible MLflow entry points
- `python_env.yaml`: MLflow Python environment spec
- `requirements.txt`: shared Python dependencies

## Recommended Experiment Strategy

The main reproducible experiment for this project is a three-model comparison:

- `lr`: logistic regression
- `rf`: random forest
- `gbt`: gradient-boosted trees

The command below uses:

- years `>= 2022`
- sampled train/valid/test splits
- MLflow logging
- automatic best-model export
- best-model tags in MLflow such as `selection_status=champion`

This is the recommended default because it is realistic for a local laptop.

## 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Configure Paths

Use environment variables instead of personal absolute paths:

```bash
export WORKSPACE_ROOT="/path/to/Group_Project"
export PROJECT_ROOT="$WORKSPACE_ROOT/DATA228_GroupProject"
export PYSPARK_PYTHON="$PROJECT_ROOT/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PROJECT_ROOT/.venv/bin/python"
export GOOGLE_APPLICATION_CREDENTIALS="$WORKSPACE_ROOT/.gcloud-config/application_default_credentials.json"
export CONNECTOR_JAR="$WORKSPACE_ROOT/third_party/gcs-connector-hadoop3-2.2.30-shaded.jar"
```

## 3. Prepare Feature Data

If you want a local copy of the prepared feature layer:

```bash
cd "$PROJECT_ROOT"
gcloud storage cp -r gs://data228/feature/bts_delay_15 data/feature/
```

After that, you can use either:

- `gs://data228/feature/bts_delay_15`
- `data/feature/bts_delay_15`

## 4. Run the Main Three-Model Comparison

This is the recommended command for local testing and demo results.

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate

./run_spark_with_gcs.sh \
  --conf spark.ui.showConsoleProgress=true \
  train_bts_delay_mlflow.py \
  --input gs://data228/feature/bts_delay_15 \
  --tracking-uri file:./mlruns \
  --experiment-name bts-delay-recent-3models-sampled \
  --models lr,rf,gbt \
  --min-year 2022 \
  --primary-metric valid_pr_auc \
  --train-sample-frac 0.01 \
  --valid-sample-frac 0.02 \
  --test-sample-frac 0.02 \
  --best-model-output data/model/bts_delay_best_recent_3models_sampled \
  --overwrite-best-model
```

What this run does:

- keeps the experiment focused on recent data
- reduces the split sizes to a laptop-friendly scale
- compares all three candidate models under the same feature pipeline
- logs one parent run plus one child run for each model
- marks the selected best child run with:
  - `selection_status=champion`
  - `is_best_model=true`
- marks other child runs with:
  - `selection_status=challenger`
  - `is_best_model=false`

## 5. Why This Command Uses Sampling

On my machine, full-data tree-based runs can fail with Spark JVM memory errors,
especially during `rf` and `gbt` evaluation. Because of that, the shared default
command in this guide uses:

- `--min-year 2022`
- `--train-sample-frac 0.01`
- `--valid-sample-frac 0.02`
- `--test-sample-frac 0.02`

This still produces a meaningful multi-model comparison while staying practical
for local development.

## 6. How to Change the Range or Scale

If another machine has more memory, these are the main knobs to change.

### Change the year range

- use `--min-year 2022` to keep only recent data
- omit `--min-year` to use all available years
- use `--max-year` together with `--min-year` if you want a bounded window

Examples:

```bash
--min-year 2022
```

```bash
--min-year 2021 --max-year 2023
```

```bash
# use the full year range
# remove both --min-year and --max-year
```

### Change the sample size

Larger fractions mean more data and more memory usage.

Examples:

```bash
--train-sample-frac 0.01 --valid-sample-frac 0.02 --test-sample-frac 0.02
```

```bash
--train-sample-frac 0.10 --valid-sample-frac 0.10 --test-sample-frac 0.10
```

```bash
--train-sample-frac 1.0 --valid-sample-frac 1.0 --test-sample-frac 1.0
```

## 7. Full-Dataset Command for Stronger Machines

If a machine has enough memory, this command runs the full three-model comparison
without year filtering or sampling:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate

./run_spark_with_gcs.sh \
  --conf spark.ui.showConsoleProgress=true \
  train_bts_delay_mlflow.py \
  --input gs://data228/feature/bts_delay_15 \
  --tracking-uri file:./mlruns \
  --experiment-name bts-delay-full-3models \
  --models lr,rf,gbt \
  --primary-metric valid_pr_auc \
  --train-sample-frac 1.0 \
  --valid-sample-frac 1.0 \
  --test-sample-frac 1.0 \
  --best-model-output data/model/bts_delay_best_full_3models \
  --overwrite-best-model
```

If this fails with `java.lang.OutOfMemoryError`, the recommended fallback is to
return to the sampled recent-years command from Section 4.

## 8. Open MLflow

To view the recorded runs:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Then open:

```text
http://127.0.0.1:5001
```

## 9. How to Read the MLflow UI

Open the experiment used by the command, for example:

- `bts-delay-recent-3models-sampled`

The run structure is:

- one parent run: `bts-delay-model-selection`
- three child runs: `lr`, `rf`, `gbt`

### To compare the three model runs

1. Open the experiment's `Runs` page.
2. Find the parent run `bts-delay-model-selection`.
3. Click the blue `+` icon to expand nested child runs.
4. Open the child runs `lr`, `rf`, and `gbt`, or use the run list to select them for comparison.

Useful places to look in the run UI:

- parent run `Overview`
  - overall experiment parameters
  - `best_model_name`
  - `best_model_run_id`
  - `best_model_exported`
- child run `Overview`
  - model-specific params such as tree depth or logistic regression settings
  - model-specific metrics
  - tags such as `selection_status`

### Best model information in the parent run

In the parent run:

- `Overview -> Parameters`
  - `best_model_name`
  - `best_model_run_id`
  - `models`
  - `min_year`
  - sampling fractions
- `Overview -> Tags`
  - `best_model_exported`
  - `best_model_name`
  - `best_model_run_id`

### Champion and challenger information in child runs

In the best child run:

- `Overview -> Tags`
  - `selection_status=champion`
  - `is_best_model=true`

In the other child runs:

- `Overview -> Tags`
  - `selection_status=challenger`
  - `is_best_model=false`

### Which metrics to compare across child runs

The most useful metrics are:

- `valid_pr_auc`
- `valid_roc_auc`
- `valid_f1`
- `test_pr_auc`
- `test_roc_auc`
- `test_f1`

If your MLflow UI supports run comparison, compare the `lr`, `rf`, and `gbt`
child runs directly. If not, open each child run and compare the metric values
manually.

### Model Registry and aliases

If the training command includes:

- `--registered-model-name`

then each candidate child run will also be registered as a new version under a
single registered model in `Model registry`.

Example:

- registered model: `bts-delay-predictor`
- candidate versions: one version each for `lr`, `rf`, and `gbt`
- best version alias: `@champion`

Where to look:

1. Open `Model registry` in the left sidebar.
2. Open the registered model name you passed with `--registered-model-name`.
3. In the `Versions` table:
   - each version corresponds to one candidate model
   - the `Aliases` column shows `@champion` on the best version
4. Open a version to inspect:
   - version tags such as `model_name`
   - `selection_status=champion` or `challenger`
   - `source_run_id`

### Where the exported best model goes

The selected model is written to the directory passed in:

- `--best-model-output`

For the main sampled experiment in this guide, that is:

- `data/model/bts_delay_best_recent_3models_sampled`

## 10. Open the Prediction UI

After you start the API with the exported best model:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
MODEL_PATH=data/model/bts_delay_best_recent_3models_sampled python -m uvicorn bts_delay_api:app --reload
```

Open one of these pages in the browser:

- `http://127.0.0.1:8000/app`
  - original single-model UI kept from the repository
  - shows the baseline page and static single-model labels
- `http://127.0.0.1:8000/app_multimodel`
  - multi-model UI for the MLflow workflow
  - reads the deployed best-model metadata from `/app_config`
  - shows labels such as `Selected Model`, `GBT Best`, and `Gradient-Boosted Trees`

Recommended demo flow:

1. Open MLflow at `http://127.0.0.1:5001`.
2. Show the three-model comparison and the selected best run.
3. Open `http://127.0.0.1:8000/app_multimodel`.
4. Click `Use Sample Flight` or fill in a flight and then click `Predict Delay`.
5. Show that the UI is using the exported best model rather than the original single-model baseline page.

## 11. Use MLproject (Optional) for Reproducibility

`MLproject` is the reproducible entry point for rerunning the same workflow in a
standardized way.

Example:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate

mlflow run . -e train_experiments \
  -P input=data/feature/bts_delay_15 \
  -P tracking_uri=file:./mlruns \
  -P experiment_name=bts-delay-reproduce-local \
  -P models=lr,rf,gbt \
  -P primary_metric=valid_pr_auc \
  -P train_sample_frac=0.01 \
  -P valid_sample_frac=0.02 \
  -P test_sample_frac=0.02 \
  -P best_model_output=data/model/bts_delay_best_reproduce_local
```

Note:

- `MLproject` does not currently expose `min_year`
- because of that, the exact `--min-year 2022` run from Section 4 is best run
  directly through `train_bts_delay_mlflow.py`
- `mlflow run` is still useful to show reproducible reruns with the same
  environment and parameters on a local feature copy

## 12. Troubleshooting

- `min_year: None` and `max_year: None` mean no year filter was used
- `query_trace_metrics is not supported with FileStore` is an MLflow UI warning
  related to trace features, not usually a failure of the experiment itself
- `java.lang.OutOfMemoryError: Java heap space` during `rf` or `gbt` usually means
  the current command is too large for local memory
- if that happens, reduce:
  - the year range with `--min-year` / `--max-year`
  - the sample fractions

## 13. Suggested Report Framing

You can present the work like this:

- the project uses MLflow for reproducible experiment tracking
- the main comparison evaluates `lr`, `rf`, and `gbt` under the same feature pipeline
- the best model is selected by validation PR AUC
- the selected best model is tagged in MLflow and exported for deployment
- the same model artifact can be served through the FastAPI app
