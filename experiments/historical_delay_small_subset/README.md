# Historical Delay Feature Exploration (Small-Subset Experiment)

This folder contains exploratory work for historical delay-rate feature engineering.  
It is **not part of the main production-like pipeline** and is preserved as a small-scale experimental branch of the project.

## Purpose
This experiment was created to evaluate whether adding historical delay statistics could improve flight delay prediction performance.

## Experimental features
- `origin_hist_delay_rate`
- `route_hist_delay_rate`

## Scope
Because full-data historical feature generation exceeded local disk limits during shuffle-heavy Spark processing, this experiment was conducted on a constrained subset:
- years 2022–2024 only
- subset sampling during feature generation
- preserved temporal split logic for train / valid / test

## Files
- `build_bts_delay_features_hist_v1.py`  
  Experimental feature-engineering script with historical delay-rate features

- `build_bts_delay_features_small_baseline.py`  
  Controlled small-subset baseline script without historical features

- `xu_handoff_notes.md`  
  Notes on setup, environment issues, experiment outputs, and recommended next steps

- `model_evaluation.md`  
  Baseline results, historical-feature results, and controlled comparison summary

## Status
This work is preserved as **exploratory / future work**.  
The main implementation priority of the project remains:
1. reproducibility improvements
2. frontend explanation and visualization improvements

## Notes
This folder is intended to document technical exploration without affecting the main project pipeline.