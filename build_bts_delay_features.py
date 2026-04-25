#!/usr/bin/env python3
"""Build a model-ready feature dataset for BTS arrival delay >= 15 minute prediction.

Examples:
  spark-submit build_bts_delay_features.py \
    --input data/curated/bts_ontime \
    --output data/feature/bts_delay_15

  ./run_spark_with_gcs.sh build_bts_delay_features.py \
    --input gs://data228/bts_ontime \
    --output gs://data228/bts_delay_15_features
"""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession, functions as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a baseline feature dataset for BTS delay >= 15 minute prediction."
    )
    parser.add_argument(
        "--input",
        default="data/curated/bts_ontime",
        help="input curated parquet root, local path or gs:// path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output feature dataset root, local path or gs:// path",
    )
    parser.add_argument(
        "--app-name",
        default="BuildBtsDelayFeatures",
        help="Spark application name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the output path if it already exists",
    )
    return parser


def hhmm_to_minutes(column: str) -> F.Column:
    padded = F.lpad(F.col(column).cast("int").cast("string"), 4, "0")
    hours = F.substring(padded, 1, 2).cast("int")
    minutes = F.substring(padded, 3, 2).cast("int")
    return hours * F.lit(60) + minutes


def add_time_features(df: DataFrame) -> DataFrame:
    dep_minutes = hhmm_to_minutes("CRSDepTime")
    arr_minutes = hhmm_to_minutes("CRSArrTime")

    return (
        df.withColumn("sched_dep_minutes", dep_minutes)
        .withColumn("sched_arr_minutes", arr_minutes)
        .withColumn("sched_dep_hour", F.floor(F.col("sched_dep_minutes") / 60))
        .withColumn("sched_arr_hour", F.floor(F.col("sched_arr_minutes") / 60))
        .withColumn(
            "dep_time_bucket",
            F.when(F.col("sched_dep_hour") < 6, F.lit("overnight"))
            .when(F.col("sched_dep_hour") < 12, F.lit("morning"))
            .when(F.col("sched_dep_hour") < 18, F.lit("afternoon"))
            .otherwise(F.lit("evening")),
        )
    )


def add_calendar_features(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("flight_date", F.to_date("FlightDate"))
        .withColumn("day_of_year", F.dayofyear("flight_date"))
        .withColumn("week_of_year", F.weekofyear("flight_date"))
        .withColumn("is_weekend", F.col("DayOfWeek").isin([6, 7]).cast("int"))
        .withColumn(
            "season",
            F.when(F.col("Month").isin([12, 1, 2]), F.lit("winter"))
            .when(F.col("Month").isin([3, 4, 5]), F.lit("spring"))
            .when(F.col("Month").isin([6, 7, 8]), F.lit("summer"))
            .otherwise(F.lit("fall")),
        )
    )


def add_route_features(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("route", F.concat_ws("_", F.col("Origin"), F.col("Dest")))
        .withColumn(
            "carrier_route",
            F.concat_ws("_", F.col("Reporting_Airline"), F.col("Origin"), F.col("Dest")),
        )
    )


def add_label_and_split(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("label_delay_15", F.col("ArrDel15").cast("int"))
        .withColumn("arrival_delay_minutes", F.col("ArrDelay").cast("double"))
        .withColumn(
            "split",
            F.when(F.col("Year") <= 2022, F.lit("train"))
            .when(F.col("Year") == 2023, F.lit("valid"))
            .otherwise(F.lit("test")),
        )
    )


def select_feature_columns(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("Year").cast("int").alias("year"),
        F.col("Quarter").cast("int").alias("quarter"),
        F.col("Month").cast("int").alias("month"),
        F.col("DayofMonth").cast("int").alias("day_of_month"),
        F.col("DayOfWeek").cast("int").alias("day_of_week"),
        F.col("flight_date"),
        F.col("day_of_year").cast("int"),
        F.col("week_of_year").cast("int"),
        F.col("is_weekend"),
        F.col("season"),
        F.col("sched_dep_minutes").cast("int"),
        F.col("sched_arr_minutes").cast("int"),
        F.col("sched_dep_hour").cast("int"),
        F.col("sched_arr_hour").cast("int"),
        F.col("dep_time_bucket"),
        F.col("Reporting_Airline").alias("carrier"),
        F.col("Origin").alias("origin"),
        F.col("OriginState").alias("origin_state"),
        F.col("OriginWac").cast("int").alias("origin_wac"),
        F.col("Dest").alias("dest"),
        F.col("DestState").alias("dest_state"),
        F.col("DestWac").cast("int").alias("dest_wac"),
        F.col("route"),
        F.col("carrier_route"),
        F.col("CRSElapsedTime").cast("double").alias("scheduled_elapsed_time"),
        F.col("Distance").cast("double").alias("distance"),
        F.col("DistanceGroup").cast("int").alias("distance_group"),
        F.col("DOT_ID_Reporting_Airline").cast("int").alias("dot_id_reporting_airline"),
        F.col("OriginAirportID").cast("int").alias("origin_airport_id"),
        F.col("DestAirportID").cast("int").alias("dest_airport_id"),
        F.col("label_delay_15"),
        F.col("arrival_delay_minutes"),
        F.col("split"),
    )


def build_features(df: DataFrame) -> DataFrame:
    filtered = df.filter(
        (F.col("Cancelled") == 0)
        & (F.col("Diverted") == 0)
        & F.col("ArrDel15").isNotNull()
        & F.col("ArrDelay").isNotNull()
        & F.col("CRSDepTime").isNotNull()
        & F.col("CRSArrTime").isNotNull()
        & F.col("CRSElapsedTime").isNotNull()
        & F.col("Distance").isNotNull()
        & F.col("Reporting_Airline").isNotNull()
        & F.col("Origin").isNotNull()
        & F.col("Dest").isNotNull()
    )

    transformed = add_time_features(filtered)
    transformed = add_calendar_features(transformed)
    transformed = add_route_features(transformed)
    transformed = add_label_and_split(transformed)
    return select_feature_columns(transformed)


def main() -> int:
    args = build_parser().parse_args()

    spark = SparkSession.builder.appName(args.app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(args.input)
        feature_df = build_features(df)

        row_count = feature_df.count()
        print(f"input: {args.input}")
        print(f"output: {args.output}")
        print(f"rows: {row_count}")
        print(f"columns: {len(feature_df.columns)}")
        print(feature_df.columns)
        feature_df.groupBy("split").count().orderBy("split").show(truncate=False)

        writer = (
            feature_df.write.mode("overwrite" if args.overwrite else "errorifexists")
            .option("compression", "snappy")
            .partitionBy("split", "year", "month")
        )
        writer.parquet(args.output)
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
