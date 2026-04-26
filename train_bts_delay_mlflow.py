#!/usr/bin/env python3
"""Tracked BTS delay model experiments with MLflow and Spark.

This script extends the baseline training flow in two ways:
1. it logs parameters, metrics, artifacts, and models to MLflow
2. it compares multiple candidate models instead of training only one baseline

Examples:
  python train_bts_delay_mlflow.py \
    --input data/feature/bts_delay_15 \
    --tracking-uri file:./mlruns \
    --experiment-name bts-delay-demo \
    --models lr,rf \
    --best-model-output data/model/bts_delay_best \
    --overwrite-best-model

  ./run_spark_with_gcs.sh train_bts_delay_mlflow.py \
    --input gs://data228/feature/bts_delay_15 \
    --tracking-uri file:./mlruns \
    --experiment-name bts-delay-gcs \
    --models lr,rf,gbt \
    --best-model-output data/model/bts_delay_best \
    --overwrite-best-model
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.spark
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.storagelevel import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F


CATEGORICAL_COLS = [
    "season",
    "dep_time_bucket",
    "carrier",
    "origin",
    "dest",
]

NUMERIC_COLS = [
    "year",
    "quarter",
    "month",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "sched_dep_minutes",
    "sched_arr_minutes",
    "sched_dep_hour",
    "sched_arr_hour",
    "origin_wac",
    "dest_wac",
    "scheduled_elapsed_time",
    "distance",
    "distance_group",
]

SUPPORTED_MODELS = {"lr", "rf", "gbt"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare BTS delay models with MLflow tracking for "
            "reproducible experiments."
        )
    )
    parser.add_argument(
        "--input",
        default="data/feature/bts_delay_15",
        help="input feature parquet root, local path or gs:// path",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        default="bts-delay-experiments",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--models",
        default="lr,rf",
        help="comma-separated model list from: lr, rf, gbt",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="optional lower bound on year to reduce the training window",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="optional upper bound on year to reduce the training window",
    )
    parser.add_argument(
        "--primary-metric",
        default="valid_pr_auc",
        choices=[
            "valid_pr_auc",
            "valid_roc_auc",
            "valid_f1",
            "test_pr_auc",
            "test_roc_auc",
            "test_f1",
        ],
        help="metric used to choose the best model",
    )
    parser.add_argument(
        "--best-model-output",
        default="data/model/bts_delay_best",
        help="directory where the best model will be exported",
    )
    parser.add_argument(
        "--overwrite-best-model",
        action="store_true",
        help="overwrite the exported best-model directory if it already exists",
    )
    parser.add_argument(
        "--train-sample-frac",
        type=float,
        default=0.01,
        help="fraction of the training split to use",
    )
    parser.add_argument(
        "--valid-sample-frac",
        type=float,
        default=0.05,
        help="fraction of the validation split to use",
    )
    parser.add_argument(
        "--test-sample-frac",
        type=float,
        default=0.05,
        help="fraction of the test split to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for sampling and model training",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="max iterations for logistic regression",
    )
    parser.add_argument(
        "--reg-param",
        type=float,
        default=0.0,
        help="regularization parameter for logistic regression",
    )
    parser.add_argument(
        "--elastic-net-param",
        type=float,
        default=0.0,
        help="elastic net mixing parameter for logistic regression",
    )
    parser.add_argument(
        "--rf-num-trees",
        type=int,
        default=80,
        help="number of trees for random forest",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=10,
        help="max depth for random forest",
    )
    parser.add_argument(
        "--gbt-max-iter",
        type=int,
        default=40,
        help="max boosting iterations for gradient-boosted trees",
    )
    parser.add_argument(
        "--gbt-max-depth",
        type=int,
        default=6,
        help="max tree depth for gradient-boosted trees",
    )
    parser.add_argument(
        "--app-name",
        default="TrainBtsDelayMlflow",
        help="Spark application name",
    )
    return parser


def parse_models(value: str) -> list[str]:
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model must be provided.")

    invalid = [model for model in models if model not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(
            f"Unsupported model(s): {', '.join(sorted(invalid))}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_MODELS))}."
        )
    return models


def path_exists(spark: SparkSession, path: str) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hconf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def remove_path(spark: SparkSession, path: str) -> None:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hconf)
    fs.delete(jvm.org.apache.hadoop.fs.Path(path), True)


def build_feature_stages() -> list[Any]:
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_idx",
            handleInvalid="keep",
        )
        for column in CATEGORICAL_COLS
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{column}_idx" for column in CATEGORICAL_COLS],
        outputCols=[f"{column}_oh" for column in CATEGORICAL_COLS],
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=NUMERIC_COLS + [f"{column}_oh" for column in CATEGORICAL_COLS],
        outputCol="features",
        handleInvalid="keep",
    )

    return indexers + [encoder, assembler]


def build_estimator(model_name: str, args: argparse.Namespace):
    if model_name == "lr":
        return LogisticRegression(
            featuresCol="features",
            labelCol="label_delay_15",
            maxIter=args.max_iter,
            regParam=args.reg_param,
            elasticNetParam=args.elastic_net_param,
        )

    if model_name == "rf":
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="label_delay_15",
            numTrees=args.rf_num_trees,
            maxDepth=args.rf_max_depth,
            seed=args.seed,
        )

    if model_name == "gbt":
        return GBTClassifier(
            featuresCol="features",
            labelCol="label_delay_15",
            maxIter=args.gbt_max_iter,
            maxDepth=args.gbt_max_depth,
            seed=args.seed,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str, args: argparse.Namespace) -> Pipeline:
    estimator = build_estimator(model_name, args)
    return Pipeline(stages=build_feature_stages() + [estimator])


def maybe_sample(split_df: DataFrame, frac: float, seed: int) -> DataFrame:
    if frac < 1.0:
        return split_df.sample(withReplacement=False, fraction=frac, seed=seed)
    return split_df


def collect_metrics(predictions: DataFrame, split_name: str) -> dict[str, float]:
    binary_eval = BinaryClassificationEvaluator(
        labelCol="label_delay_15",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol="label_delay_15",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label_delay_15",
        predictionCol="prediction",
        metricName="f1",
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label_delay_15",
        predictionCol="prediction",
        metricName="weightedPrecision",
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="label_delay_15",
        predictionCol="prediction",
        metricName="weightedRecall",
    )

    return {
        f"{split_name}_roc_auc": float(binary_eval.evaluate(predictions)),
        f"{split_name}_pr_auc": float(pr_eval.evaluate(predictions)),
        f"{split_name}_f1": float(f1_eval.evaluate(predictions)),
        f"{split_name}_precision": float(precision_eval.evaluate(predictions)),
        f"{split_name}_recall": float(recall_eval.evaluate(predictions)),
    }


def preview_predictions(predictions: DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    rows = (
        predictions.select(
            "label_delay_15",
            "prediction",
            vector_to_array("probability")[1].alias("probability_delay_15"),
            "carrier",
            "origin",
            "dest",
            "split",
        )
        .limit(limit)
        .collect()
    )
    return [row.asDict(recursive=True) for row in rows]


def log_json_artifact(payload: Any, artifact_file: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / Path(artifact_file).name
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_path), artifact_path=str(Path(artifact_file).parent))


def model_params(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    if model_name == "lr":
        return {
            "model_type": "logistic_regression",
            "max_iter": args.max_iter,
            "reg_param": args.reg_param,
            "elastic_net_param": args.elastic_net_param,
        }
    if model_name == "rf":
        return {
            "model_type": "random_forest",
            "rf_num_trees": args.rf_num_trees,
            "rf_max_depth": args.rf_max_depth,
        }
    if model_name == "gbt":
        return {
            "model_type": "gradient_boosted_trees",
            "gbt_max_iter": args.gbt_max_iter,
            "gbt_max_depth": args.gbt_max_depth,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> int:
    args = build_parser().parse_args()
    selected_models = parse_models(args.models)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    spark = SparkSession.builder.appName(args.app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(args.input)
        if args.min_year is not None:
            df = df.filter(F.col("year") >= args.min_year)
        if args.max_year is not None:
            df = df.filter(F.col("year") <= args.max_year)

        train_df = df.filter(F.col("split") == "train")
        valid_df = df.filter(F.col("split") == "valid")
        test_df = df.filter(F.col("split") == "test")

        sampled_train_df = maybe_sample(train_df, args.train_sample_frac, args.seed)
        sampled_valid_df = maybe_sample(valid_df, args.valid_sample_frac, args.seed + 1)
        sampled_test_df = maybe_sample(test_df, args.test_sample_frac, args.seed + 2)

        sampled_train_rows = sampled_train_df.count()
        sampled_valid_rows = sampled_valid_df.count()
        sampled_test_rows = sampled_test_df.count()

        print(f"input: {args.input}")
        print(f"tracking_uri: {args.tracking_uri}")
        print(f"experiment_name: {args.experiment_name}")
        print(f"models: {selected_models}")
        print(f"min_year: {args.min_year}")
        print(f"max_year: {args.max_year}")
        print(f"sampled_train_rows: {sampled_train_rows}")
        print(f"sampled_valid_rows: {sampled_valid_rows}")
        print(f"sampled_test_rows: {sampled_test_rows}")

        leaderboard: list[dict[str, Any]] = []
        best_model = None
        best_model_name = None
        best_score = float("-inf")
        best_child_run_id = None

        with mlflow.start_run(run_name="bts-delay-model-selection") as parent_run:
            mlflow.set_tags(
                {
                    "project": "data228-bts-delay",
                    "pipeline": "spark+mlflow",
                    "goal": "reproducible_model_selection",
                }
            )
            mlflow.log_params(
                {
                    "input": args.input,
                    "models": ",".join(selected_models),
                    "primary_metric": args.primary_metric,
                    "min_year": "" if args.min_year is None else args.min_year,
                    "max_year": "" if args.max_year is None else args.max_year,
                    "train_sample_frac": args.train_sample_frac,
                    "valid_sample_frac": args.valid_sample_frac,
                    "test_sample_frac": args.test_sample_frac,
                    "seed": args.seed,
                }
            )
            mlflow.log_metrics(
                {
                    "sampled_train_rows": float(sampled_train_rows),
                    "sampled_valid_rows": float(sampled_valid_rows),
                    "sampled_test_rows": float(sampled_test_rows),
                }
            )
            log_json_artifact(
                {
                    "categorical_cols": CATEGORICAL_COLS,
                    "numeric_cols": NUMERIC_COLS,
                    "selected_models": selected_models,
                },
                "config/feature_schema.json",
            )

            for model_name in selected_models:
                print(f"\n=== training {model_name} ===")
                with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                    mlflow.log_params(model_params(model_name, args))
                    pipeline = build_pipeline(model_name, args)
                    model = pipeline.fit(sampled_train_df)

                    valid_predictions = model.transform(sampled_valid_df).persist(
                        StorageLevel.DISK_ONLY
                    )
                    test_predictions = model.transform(sampled_test_df).persist(
                        StorageLevel.DISK_ONLY
                    )

                    metrics = {
                        **collect_metrics(valid_predictions, "valid"),
                        **collect_metrics(test_predictions, "test"),
                    }
                    mlflow.log_metrics(metrics)
                    mlflow.spark.log_model(model, artifact_path="model")
                    log_json_artifact(
                        preview_predictions(valid_predictions),
                        f"preview/{model_name}_valid_predictions.json",
                    )

                    score = metrics[args.primary_metric]
                    leaderboard_entry = {
                        "model_name": model_name,
                        "run_id": child_run.info.run_id,
                        "selection_score": score,
                        **metrics,
                    }
                    leaderboard.append(leaderboard_entry)

                    for metric_name, metric_value in metrics.items():
                        print(f"{model_name}_{metric_name}: {metric_value:.6f}")

                    valid_predictions.unpersist()
                    test_predictions.unpersist()

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_name = model_name
                        best_child_run_id = child_run.info.run_id

            leaderboard.sort(key=lambda item: item["selection_score"], reverse=True)
            mlflow.log_metric(f"best_{args.primary_metric}", best_score)
            mlflow.log_params(
                {
                    "best_model_name": best_model_name,
                    "best_model_run_id": best_child_run_id or "",
                }
            )
            log_json_artifact(leaderboard, "reports/leaderboard.json")

            print("\n=== leaderboard ===")
            for rank, item in enumerate(leaderboard, start=1):
                print(
                    f"{rank}. {item['model_name']} "
                    f"{args.primary_metric}={item['selection_score']:.6f} "
                    f"run_id={item['run_id']}"
                )

            if best_model is None or best_model_name is None:
                raise RuntimeError("No model was trained successfully.")

            if path_exists(spark, args.best_model_output):
                if args.overwrite_best_model:
                    remove_path(spark, args.best_model_output)
                else:
                    raise RuntimeError(
                        f"best model output already exists: {args.best_model_output} "
                        "(pass --overwrite-best-model to replace it)"
                    )

            best_model.write().save(args.best_model_output)
            mlflow.log_param("best_model_output", args.best_model_output)
            mlflow.set_tag("best_model_exported", "true")

            print("\n=== best model ===")
            print(f"parent_run_id: {parent_run.info.run_id}")
            print(f"best_model_name: {best_model_name}")
            print(f"best_model_run_id: {best_child_run_id}")
            print(f"{args.primary_metric}: {best_score:.6f}")
            print(f"best_model_output: {args.best_model_output}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
