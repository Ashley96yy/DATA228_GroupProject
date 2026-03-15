#!/usr/bin/env python3
"""Simple Spark read test for the curated BTS dataset."""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a curated BTS Parquet path with Spark and print a small preview."
    )
    parser.add_argument(
        "--path",
        default="gs://data228/bts_ontime/year=2024/month=01/",
        help="Parquet path to read, local or gs://",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="number of preview rows to show",
    )
    parser.add_argument(
        "--app-name",
        default="TestReadBtsParquet",
        help="Spark application name",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    spark = SparkSession.builder.appName(args.app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(args.path)
        print(f"path: {args.path}")
        print(f"columns: {len(df.columns)}")
        print(df.columns[:10])

        preview_cols = [
            column
            for column in ["Year", "Month", "Origin", "Dest", "DepDelay", "ArrDelay"]
            if column in df.columns
        ]

        if preview_cols:
            df.select(*preview_cols).show(args.rows, truncate=False)
        else:
            df.show(args.rows, truncate=False)
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
