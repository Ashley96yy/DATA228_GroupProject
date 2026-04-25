#!/usr/bin/env python3
"""Train a baseline Spark ML model for BTS delay >= 15 minute prediction.

Examples:
  spark-submit train_bts_delay_model.py \
    --input data/feature/bts_delay_15 \
    --output-dir data/model/bts_delay_lr_baseline \
    --train-sample-frac 0.02

  ./run_spark_with_gcs.sh train_bts_delay_model.py \
    --input gs://data228/bts_delay_15_features \
    --output-dir gs://data228/models/bts_delay_lr_baseline \
    --train-sample-frac 0.02
"""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession, functions as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a baseline logistic regression model for BTS delay prediction."
    )
    parser.add_argument(
        "--input",
        default="data/feature/bts_delay_15",
        help="input feature parquet root, local path or gs:// path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/model/bts_delay_lr_baseline",
        help="directory where the trained pipeline model will be saved",
    )
    parser.add_argument(
        "--train-sample-frac",
        type=float,
        default=0.01,
        help="fraction of the training split to use for baseline training",
    )
    parser.add_argument(
        "--valid-sample-frac",
        type=float,
        default=0.05,
        help="fraction of the validation split to use for local baseline evaluation",
    )
    parser.add_argument(
        "--test-sample-frac",
        type=float,
        default=0.05,
        help="fraction of the test split to use for local baseline evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for sampling",
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
        "--overwrite",
        action="store_true",
        help="overwrite the saved model output directory",
    )
    parser.add_argument(
        "--app-name",
        default="TrainBtsDelayModel",
        help="Spark application name",
    )
    return parser


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


def build_pipeline(max_iter: int, reg_param: float, elastic_net_param: float) -> Pipeline:
    categorical_cols = [
        "season",
        "dep_time_bucket",
        "carrier",
        "origin",
        "dest",
    ]
    numeric_cols = [
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

    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_idx",
            handleInvalid="keep",
        )
        for column in categorical_cols
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{column}_idx" for column in categorical_cols],
        outputCols=[f"{column}_oh" for column in categorical_cols],
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{column}_oh" for column in categorical_cols],
        outputCol="features",
        handleInvalid="keep",
    )

    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label_delay_15",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
    )

    return Pipeline(stages=indexers + [encoder, assembler, classifier])


def evaluate_predictions(predictions, split_name: str) -> None:
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

    print(f"{split_name}_roc_auc: {binary_eval.evaluate(predictions):.6f}")
    print(f"{split_name}_pr_auc: {pr_eval.evaluate(predictions):.6f}")
    print(f"{split_name}_f1: {f1_eval.evaluate(predictions):.6f}")
    print(f"{split_name}_precision: {precision_eval.evaluate(predictions):.6f}")
    print(f"{split_name}_recall: {recall_eval.evaluate(predictions):.6f}")


def main() -> int:
    args = build_parser().parse_args()

    spark = SparkSession.builder.appName(args.app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(args.input)

        train_df = df.filter(F.col("split") == "train")
        valid_df = df.filter(F.col("split") == "valid")
        test_df = df.filter(F.col("split") == "test")

        def maybe_sample(split_df, frac: float, seed_offset: int):
            if frac < 1.0:
                return split_df.sample(
                    withReplacement=False,
                    fraction=frac,
                    seed=args.seed + seed_offset,
                )
            return split_df

        sampled_train_df = maybe_sample(train_df, args.train_sample_frac, 0)
        sampled_valid_df = maybe_sample(valid_df, args.valid_sample_frac, 1)
        sampled_test_df = maybe_sample(test_df, args.test_sample_frac, 2)

        print(f"input: {args.input}")
        print(f"output_dir: {args.output_dir}")
        print(f"train_sample_frac: {args.train_sample_frac}")
        print(f"valid_sample_frac: {args.valid_sample_frac}")
        print(f"test_sample_frac: {args.test_sample_frac}")
        print(f"sampled_train_rows: {sampled_train_df.count()}")
        print(f"sampled_valid_rows: {sampled_valid_df.count()}")
        print(f"sampled_test_rows: {sampled_test_df.count()}")

        sampled_train_df.groupBy("label_delay_15").count().orderBy("label_delay_15").show()

        pipeline = build_pipeline(
            max_iter=args.max_iter,
            reg_param=args.reg_param,
            elastic_net_param=args.elastic_net_param,
        )
        model = pipeline.fit(sampled_train_df)

        valid_predictions = model.transform(sampled_valid_df).persist(StorageLevel.DISK_ONLY)
        test_predictions = model.transform(sampled_test_df).persist(StorageLevel.DISK_ONLY)

        evaluate_predictions(valid_predictions, "valid")
        evaluate_predictions(test_predictions, "test")

        valid_predictions.select(
            "label_delay_15", "prediction", "probability"
        ).show(5, truncate=False)

        if path_exists(spark, args.output_dir):
            if args.overwrite:
                remove_path(spark, args.output_dir)
            else:
                raise RuntimeError(
                    f"output path already exists: {args.output_dir} "
                    "(pass --overwrite to replace it)"
                )

        model.write().save(args.output_dir)
        print(f"saved_model: {args.output_dir}")
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
