# Model Evaluation Notes

## Summary
This document records three stages of model evaluation:
1. A baseline local training run on the existing feature dataset
2. An exploratory historical-feature experiment under local resource constraints
3. A controlled small-subset comparison between the small baseline and the historical-feature variant

Overall, the historical-delay features were technically feasible and trainable, but they did not show consistent performance improvement across all metrics in the constrained comparison.

## Baseline local training run
Model output: `data/model/my_lr_test`

### Dataset sampling
- train_sample_frac: 0.002
- valid_sample_frac: 0.01
- test_sample_frac: 0.01

### Label distribution (sampled training subset)
- label 0: 330819
- label 1: 80859

### Validation metrics
- ROC AUC: 0.624767
- PR AUC: 0.284460
- F1: 0.701918
- Precision: 0.676994
- Recall: 0.792960

### Test metrics
- ROC AUC: 0.629408
- PR AUC: 0.296898
- F1: 0.693157
- Precision: 0.671786
- Recall: 0.786589

## Initial observations
- The baseline pipeline runs successfully on local Spark.
- Validation and test metrics are close, suggesting stable generalization.
- The dataset is imbalanced, so PR AUC is an important metric.
- This run can serve as the comparison baseline for future feature-engineering improvements.

## Historical-feature experiment (resource-constrained subset)

### New features added
- `origin_hist_delay_rate`
- `route_hist_delay_rate`

### Experiment setup
To avoid local disk overflow during full-data shuffle operations, I created an experimental feature-engineering script and rebuilt features on a constrained subset covering years 2022–2024 with sampling. This preserved the train/valid/test split logic while making the experiment feasible on a local machine.

Enhanced feature dataset:
- `data/feature/bts_delay_15_hist_v1_small`

Enhanced model output:
- `data/model/my_lr_hist_v1_small`

### Hist_v1_small metrics
- Valid ROC AUC: 0.551856
- Valid PR AUC: 0.230830
- Valid F1: 0.715287
- Test ROC AUC: 0.548895
- Test PR AUC: 0.240555
- Test F1: 0.704996

### Comparison vs baseline
Compared with the earlier baseline local run, the constrained historical-feature experiment did not improve ROC AUC or PR AUC overall, although F1 was slightly higher. Because the historical-feature experiment used a smaller resource-constrained subset and a different feature-generation path, this comparison should be interpreted as a feasibility and pipeline-validation result rather than a final like-for-like performance comparison.

### Takeaways
- The historical-feature engineering pipeline works end-to-end.
- The added features are successfully generated and consumed by model training.
- Under constrained local resources, performance gains were not consistently observed.
- A better next step is to run a larger controlled comparison with matched sampling and training conditions.

## Controlled small-subset comparison

### Controlled comparison setup
Both `small_baseline` and `hist_v1_small` were built from the same constrained subset logic:
- years 2022–2024 only
- subset sampling applied during feature generation
- identical train/valid/test temporal split logic
- identical downstream training sample fractions

This comparison isolates the effect of adding historical airport- and route-level delay-rate features under the same small-subset setting.

| Model | Valid ROC AUC | Valid PR AUC | Valid F1 | Test ROC AUC | Test PR AUC | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| small_baseline | 0.544963 | 0.238817 | 0.698587 | 0.560772 | 0.236121 | 0.705233 |
| hist_v1_small | 0.551856 | 0.230830 | 0.715287 | 0.548895 | 0.240555 | 0.704996 |

### Interpretation
The controlled small-subset comparison showed mixed effects from the added historical-delay features. On validation data, ROC AUC and F1 improved, while PR AUC decreased. On test data, PR AUC improved slightly, ROC AUC decreased, and F1 remained nearly unchanged. These results suggest that the historical-delay features are technically feasible and worth preserving as exploratory work, but they do not yet provide a stable and consistent performance improvement under the current constrained local setup.

## Implications for next steps
The historical-feature experiment is preserved as exploratory work and as a future direction for larger-scale evaluation. Because the constrained comparison did not show consistent across-the-board improvement, the main implementation priority now shifts toward:
1. reproducibility improvements
2. frontend explanation and visualization improvements

This keeps the core pipeline stable while improving usability, interpretability, and final project presentation quality.