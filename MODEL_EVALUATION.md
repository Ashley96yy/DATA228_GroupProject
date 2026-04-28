# Model Evaluation Notes

## Baseline local training run
Model output: `data/model/my_lr_test`

### Dataset sampling
- train_sample_frac: 0.002
- valid_sample_frac: 0.01
- test_sample_frac: 0.01

### Label distribution (sampled train)
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