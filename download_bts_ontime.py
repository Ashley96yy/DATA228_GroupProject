#!/usr/bin/env python3
"""Bulk-download BTS on-time performance monthly ZIP files from PREZIP."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import socket
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from html import unescape
from http.client import IncompleteRead
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen


BASE_URL = "https://transtats.bts.gov/PREZIP/"
LISTING_FILE_RE = re.compile(
    r"""
    (?P<stamp>\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)\s+
    (?P<size>\d+)\s+
    <A\s+HREF="(?P<href>/PREZIP/)?
    (?P<filename>
        On_Time_Reporting_Carrier_On_Time_Performance
        (?:_1987_present|_\(1987_present\))
        _(?P<year>\d{4})_(?P<month>\d{1,2})\.zip
    )"
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, order=True)
class MonthlyFile:
    year: int
    month: int
    filename: str
    size_bytes: int
    listed_at: dt.datetime

    @property
    def ym(self) -> int:
        return self.year * 100 + self.month

    @property
    def url(self) -> str:
        return urljoin(BASE_URL, self.filename)


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def list_available_files() -> list[MonthlyFile]:
    html = fetch_text(BASE_URL)
    matches = LISTING_FILE_RE.finditer(unescape(html))
    by_month: dict[tuple[int, int], MonthlyFile] = {}

    for match in matches:
        file = MonthlyFile(
            year=int(match.group("year")),
            month=int(match.group("month")),
            filename=match.group("filename"),
            size_bytes=int(match.group("size")),
            listed_at=dt.datetime.strptime(match.group("stamp"), "%m/%d/%Y %I:%M %p"),
        )
        key = (file.year, file.month)
        current = by_month.get(key)
        if current is None or is_better_listing_choice(file, current):
            by_month[key] = file

    return sorted(by_month.values())


def is_better_listing_choice(candidate: MonthlyFile, current: MonthlyFile) -> bool:
    # Prefer the simpler filename variant when both exist for the same month.
    candidate_score = (
        "(" in candidate.filename,
        -candidate.size_bytes,
        -int(candidate.listed_at.timestamp()),
        candidate.filename,
    )
    current_score = (
        "(" in current.filename,
        -current.size_bytes,
        -int(current.listed_at.timestamp()),
        current.filename,
    )
    return candidate_score < current_score


def parse_ym(value: str) -> int:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise argparse.ArgumentTypeError(f"invalid YYYY-MM value: {value}")
    year, month = map(int, value.split("-"))
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"invalid month in {value}")
    return year * 100 + month


def filter_files(
    files: Iterable[MonthlyFile],
    start_ym: int,
    end_ym: int,
) -> list[MonthlyFile]:
    return [file for file in files if start_ym <= file.ym <= end_ym]


def is_valid_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def download_one(
    file: MonthlyFile,
    out_dir: Path,
    overwrite: bool,
    retries: int,
) -> str:
    dest = out_dir / file.filename
    if dest.exists() and not overwrite:
        return f"skip  {file.year}-{file.month:02d}  {dest.name}"

    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        tmp_dest.unlink(missing_ok=True)
        try:
            with urlopen(file.url, timeout=120) as response:
                total = response.headers.get("Content-Length")
                expected = int(total) if total is not None else None
                with tmp_dest.open("wb") as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            socket.timeout,
            IncompleteRead,
            OSError,
        ) as exc:
            tmp_dest.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            return (
                f"error {file.year}-{file.month:02d}  {file.filename}  "
                f"{exc} after {attempts} attempts"
            )

        actual_size = tmp_dest.stat().st_size
        size_ok = expected is None or actual_size == expected
        zip_ok = is_valid_zip(tmp_dest)
        if size_ok and zip_ok:
            tmp_dest.replace(dest)
            size_mb = actual_size / (1024 * 1024)
            return (
                f"ok    {file.year}-{file.month:02d}  {dest.name}  "
                f"{size_mb:.1f} MB"
            )

        tmp_dest.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 8))
            continue

        if not size_ok:
            return (
                f"error {file.year}-{file.month:02d}  {file.filename}  "
                f"size mismatch ({actual_size} != {expected}) after {attempts} attempts"
            )
        return (
            f"error {file.year}-{file.month:02d}  {file.filename}  "
            f"downloaded file failed ZIP validation after {attempts} attempts"
        )

    return f"error {file.year}-{file.month:02d}  {file.filename}  unknown failure"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download BTS Reporting Carrier On-Time Performance monthly ZIP files "
            "from the TranStats PREZIP directory."
        )
    )
    parser.add_argument(
        "--start",
        type=parse_ym,
        default=parse_ym("1987-10"),
        help="start month in YYYY-MM format (default: 1987-10)",
    )
    parser.add_argument(
        "--end",
        type=parse_ym,
        default=parse_ym("2025-12"),
        help="end month in YYYY-MM format (default: 2025-12)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/bts_ontime"),
        help="output directory (default: data/raw/bts_ontime)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of concurrent downloads (default: 4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download files even if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print the matched files without downloading",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of retries per file after the first failed attempt (default: 3)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be <= --end")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.retries < 0:
        parser.error("--retries must be >= 0")

    try:
        available = list_available_files()
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"failed to read PREZIP listing: {exc}", file=sys.stderr)
        return 1

    selected = filter_files(available, args.start, args.end)
    if not selected:
        print("no matching files found in PREZIP listing", file=sys.stderr)
        return 1

    latest = max(available)
    print(
        f"found {len(available)} available monthly files; "
        f"latest in listing is {latest.year}-{latest.month:02d}"
    )
    print(
        f"selected {len(selected)} files from "
        f"{selected[0].year}-{selected[0].month:02d} to "
        f"{selected[-1].year}-{selected[-1].month:02d}"
    )

    if args.dry_run:
        for file in selected:
            print(f"{file.year}-{file.month:02d}  {file.filename}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                file,
                args.out,
                args.overwrite,
                args.retries,
            )
            for file in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
