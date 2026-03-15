#!/usr/bin/env python3
"""Convert monthly BTS on-time ZIP archives into partitioned Parquet with PySpark.

Examples:
  spark-submit convert_bts_ontime_to_parquet.py \
    --input-dir data/raw/bts_ontime \
    --output data/curated/bts_ontime

  spark-submit convert_bts_ontime_to_parquet.py \
    --input-dir data/raw/bts_ontime \
    --output gs://your-bucket/bts_ontime/curated \
    --start 2019-01 \
    --end 2025-11
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession


ZIP_NAME_RE = re.compile(
    r"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_(\d{4})_(\d{1,2})\.zip$"
)


@dataclass(frozen=True, order=True)
class MonthlyZip:
    year: int
    month: int
    path: Path

    @property
    def ym(self) -> int:
        return self.year * 100 + self.month

    @property
    def output_suffix(self) -> str:
        return f"year={self.year}/month={self.month:02d}"


def parse_ym(value: str) -> int:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid YYYY-MM value: {value}")
    year = int(match.group(1))
    month = int(match.group(2))
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"invalid month in {value}")
    return year * 100 + month


def discover_monthly_zips(input_dir: Path) -> list[MonthlyZip]:
    monthly = []
    for path in sorted(input_dir.glob("On_Time_Reporting_Carrier_On_Time_Performance_1987_present_*.zip")):
        match = ZIP_NAME_RE.fullmatch(path.name)
        if not match:
            continue
        monthly.append(
            MonthlyZip(
                year=int(match.group(1)),
                month=int(match.group(2)),
                path=path,
            )
        )
    return monthly


def filter_months(files: list[MonthlyZip], start_ym: int, end_ym: int) -> list[MonthlyZip]:
    return [file for file in files if start_ym <= file.ym <= end_ym]


def path_exists(spark: SparkSession, path: str) -> bool:
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    uri = jvm.java.net.URI(path)
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hconf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def extract_csv(zip_path: Path, work_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"{zip_path.name} contains {len(csv_names)} CSV files")
        csv_name = csv_names[0]
        extracted_path = work_dir / Path(csv_name).name
        with archive.open(csv_name) as src, extracted_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return extracted_path


def convert_one_month(
    spark: SparkSession,
    monthly_zip: MonthlyZip,
    output_root: str,
    temp_root: Path,
    overwrite: bool,
) -> str:
    target = f"{output_root.rstrip('/')}/{monthly_zip.output_suffix}"
    if path_exists(spark, target) and not overwrite:
        return f"skip  {monthly_zip.year}-{monthly_zip.month:02d}  {target}"

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"bts_{monthly_zip.year}_{monthly_zip.month:02d}_",
            dir=str(temp_root),
        )
    )
    try:
        extracted_csv = extract_csv(monthly_zip.path, work_dir)
        df = (
            spark.read.option("header", True)
            .option("inferSchema", False)
            .option("mode", "FAILFAST")
            .csv(str(extracted_csv))
        )
        unnamed_columns = [column for column in df.columns if re.fullmatch(r"_c\d+", column)]
        if unnamed_columns:
            df = df.drop(*unnamed_columns)
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(target)
        )
        return f"ok    {monthly_zip.year}-{monthly_zip.month:02d}  {target}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert downloaded BTS on-time ZIP archives into partitioned Parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/bts_ontime"),
        help="directory containing the downloaded monthly ZIP files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output root directory, local path or gs:// bucket path",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/tmp/bts_ontime_extract"),
        help="temporary extraction directory",
    )
    parser.add_argument(
        "--start",
        type=parse_ym,
        default=parse_ym("1987-10"),
        help="first month to convert in YYYY-MM format",
    )
    parser.add_argument(
        "--end",
        type=parse_ym,
        default=parse_ym("2025-11"),
        help="last month to convert in YYYY-MM format",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite month partitions that already exist at the output path",
    )
    parser.add_argument(
        "--app-name",
        default="BtsOnTimeZipToParquet",
        help="Spark application name",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be <= --end")
    if not args.input_dir.exists():
        parser.error(f"--input-dir does not exist: {args.input_dir}")

    monthly_zips = discover_monthly_zips(args.input_dir)
    selected = filter_months(monthly_zips, args.start, args.end)
    if not selected:
        parser.error("no matching ZIP files were found for the requested month range")

    args.temp_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName(args.app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print(
            f"found {len(monthly_zips)} monthly ZIP files locally; "
            f"converting {len(selected)} months from "
            f"{selected[0].year}-{selected[0].month:02d} to "
            f"{selected[-1].year}-{selected[-1].month:02d}"
        )
        for monthly_zip in selected:
            print(
                convert_one_month(
                    spark=spark,
                    monthly_zip=monthly_zip,
                    output_root=args.output,
                    temp_root=args.temp_dir,
                    overwrite=args.overwrite,
                )
            )
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
