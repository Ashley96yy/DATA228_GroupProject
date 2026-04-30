# Xu handoff notes

## Current working status
- Repo cloned locally
- GCS access confirmed
- Baseline model downloaded locally
- FastAPI app runs successfully
- `/app` page loads successfully
- Local baseline training pipeline runs successfully
- Controlled small-subset feature experiments completed successfully

## Key outputs completed
- Baseline local model run completed
- Experimental historical-feature script created:
  - `build_bts_delay_features_hist_v1.py`
- Controlled small-baseline script created:
  - `build_bts_delay_features_small_baseline.py`
- Experimental feature dataset generated:
  - `data/feature/bts_delay_15_hist_v1_small`
- Controlled comparison results recorded in:
  - `MODEL_EVALUATION.md`

## Historical feature experiment
- Added two experimental historical features:
  - `origin_hist_delay_rate`
  - `route_hist_delay_rate`
- Rebuilt features on a constrained subset covering years 2022–2024 with sampling
- Successfully generated:
  - `data/feature/bts_delay_15_hist_v1_small`
- Output rows: 404650
- Output columns: 35
- Result: the controlled small-subset comparison showed mixed metric changes rather than a consistent improvement

## Issues encountered
- Local Spark could not directly read `gs://` paths without a GCS connector
- Added required GCS connector jar to `third_party/`
- `run_spark_with_gcs.sh` required Application Default Credentials (ADC) setup
- Full-data historical-feature generation exceeded local disk capacity during shuffle-heavy processing
- Used a constrained small-subset experiment to validate the feature-engineering idea under local resource limits

## Recommended next steps
1. Preserve the historical-feature experiment as exploratory work
2. Avoid changing the main production-like pipeline unless needed
3. Prioritize reproducibility improvements:
   - README cleanup
   - relative paths
   - clearer environment/setup instructions
   - minimal run path vs GCS-dependent workflow
4. Prioritize frontend explanation and visualization improvements:
   - clearer probability display
   - risk-level labels
   - prediction explanation text
   - user-facing model notes

## Main contribution direction
My main contribution focus is now shifting from feature experimentation to:
1. Reproducibility improvements
2. Frontend explanation and visualization improvements

The historical-feature small-subset experiment will be preserved as exploratory work and future-work support.