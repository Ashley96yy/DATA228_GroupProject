#!/usr/bin/env python3
"""Shared feature, prediction, and route metadata helpers for the BTS project."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


AIRPORT_METADATA: dict[str, dict[str, float | int]] = {
    "ATL": {"wac": 34, "lat": 33.6407, "lon": -84.4277, "city": "Atlanta, GA"},
    "BOS": {"wac": 13, "lat": 42.3656, "lon": -71.0096, "city": "Boston, MA"},
    "CLT": {"wac": 36, "lat": 35.2144, "lon": -80.9473, "city": "Charlotte, NC"},
    "DCA": {"wac": 38, "lat": 38.8512, "lon": -77.0402, "city": "Washington, DC"},
    "DEN": {"wac": 82, "lat": 39.8561, "lon": -104.6737, "city": "Denver, CO"},
    "DFW": {"wac": 74, "lat": 32.8998, "lon": -97.0403, "city": "Dallas/Fort Worth, TX"},
    "DTW": {"wac": 43, "lat": 42.2162, "lon": -83.3554, "city": "Detroit, MI"},
    "EWR": {"wac": 21, "lat": 40.6895, "lon": -74.1745, "city": "Newark, NJ"},
    "IAD": {"wac": 38, "lat": 38.9531, "lon": -77.4565, "city": "Washington Dulles, VA"},
    "IAH": {"wac": 74, "lat": 29.9902, "lon": -95.3368, "city": "Houston, TX"},
    "JFK": {"wac": 22, "lat": 40.6413, "lon": -73.7781, "city": "New York, NY"},
    "LAS": {"wac": 85, "lat": 36.0840, "lon": -115.1537, "city": "Las Vegas, NV"},
    "LAX": {"wac": 91, "lat": 33.9416, "lon": -118.4085, "city": "Los Angeles, CA"},
    "MCO": {"wac": 33, "lat": 28.4312, "lon": -81.3081, "city": "Orlando, FL"},
    "MIA": {"wac": 33, "lat": 25.7959, "lon": -80.2870, "city": "Miami, FL"},
    "MSP": {"wac": 63, "lat": 44.8848, "lon": -93.2223, "city": "Minneapolis, MN"},
    "ORD": {"wac": 41, "lat": 41.9742, "lon": -87.9073, "city": "Chicago, IL"},
    "PDX": {"wac": 92, "lat": 45.5898, "lon": -122.5951, "city": "Portland, OR"},
    "PHL": {"wac": 23, "lat": 39.8744, "lon": -75.2424, "city": "Philadelphia, PA"},
    "PHX": {"wac": 81, "lat": 33.4342, "lon": -112.0116, "city": "Phoenix, AZ"},
    "SAN": {"wac": 91, "lat": 32.7338, "lon": -117.1933, "city": "San Diego, CA"},
    "SAT": {"wac": 74, "lat": 29.5337, "lon": -98.4698, "city": "San Antonio, TX"},
    "SBN": {"wac": 42, "lat": 41.7087, "lon": -86.3173, "city": "South Bend, IN"},
    "SEA": {"wac": 93, "lat": 47.4502, "lon": -122.3088, "city": "Seattle, WA"},
    "SFO": {"wac": 91, "lat": 37.6213, "lon": -122.3790, "city": "San Francisco, CA"},
    "SJC": {"wac": 91, "lat": 37.3639, "lon": -121.9289, "city": "San Jose, CA"},
}


def build_flight_date(year: int, month: int, day_of_month: int) -> date:
    return date(year, month, day_of_month)


def hhmm_to_minutes(value: int) -> int:
    return (value // 100) * 60 + (value % 100)


def compute_elapsed_minutes(dep_hhmm: int, arr_hhmm: int) -> int:
    dep_minutes = hhmm_to_minutes(dep_hhmm)
    arr_minutes = hhmm_to_minutes(arr_hhmm)
    elapsed_minutes = arr_minutes - dep_minutes
    if elapsed_minutes <= 0:
        elapsed_minutes += 24 * 60
    return elapsed_minutes


def compute_quarter(month: int) -> int:
    return ((month - 1) // 3) + 1


def compute_season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "fall"


def compute_dep_time_bucket(dep_hour: int) -> str:
    if dep_hour < 6:
        return "overnight"
    if dep_hour < 12:
        return "morning"
    if dep_hour < 18:
        return "afternoon"
    return "evening"


def compute_distance_group(distance: float) -> int:
    return max(1, math.ceil(distance / 250.0))


def lookup_airport(code: str) -> dict[str, float | int]:
    airport = AIRPORT_METADATA.get(code.upper())
    if airport is None:
        raise ValueError(f"Unsupported airport code: {code}")
    return airport


def compute_route_distance(origin_code: str, dest_code: str) -> float:
    origin_code = origin_code.upper()
    dest_code = dest_code.upper()
    if origin_code == dest_code:
        raise ValueError("Origin and destination must be different.")

    origin = lookup_airport(origin_code)
    dest = lookup_airport(dest_code)

    radius_miles = 3958.8
    lat1 = math.radians(float(origin["lat"]))
    lon1 = math.radians(float(origin["lon"]))
    lat2 = math.radians(float(dest["lat"]))
    lon2 = math.radians(float(dest["lon"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(radius_miles * c, 1)


def get_route_info(origin_code: str, dest_code: str) -> dict[str, Any]:
    origin = lookup_airport(origin_code)
    dest = lookup_airport(dest_code)
    return {
        "origin": origin_code.upper(),
        "origin_city": origin["city"],
        "origin_wac": int(origin["wac"]),
        "dest": dest_code.upper(),
        "dest_city": dest["city"],
        "dest_wac": int(dest["wac"]),
        "distance_miles": compute_route_distance(origin_code, dest_code),
    }


def build_feature_row(request: dict[str, Any]) -> dict[str, int | float | str]:
    flight_dt = build_flight_date(
        int(request["year"]),
        int(request["month"]),
        int(request["day_of_month"]),
    )
    sched_dep_minutes = hhmm_to_minutes(int(request["crs_dep_time"]))
    sched_arr_minutes = hhmm_to_minutes(int(request["crs_arr_time"]))
    sched_dep_hour = sched_dep_minutes // 60
    sched_arr_hour = sched_arr_minutes // 60
    route_info = get_route_info(str(request["origin"]), str(request["dest"]))
    distance = float(route_info["distance_miles"])
    scheduled_elapsed_time = float(
        compute_elapsed_minutes(int(request["crs_dep_time"]), int(request["crs_arr_time"]))
    )

    return {
        "year": int(request["year"]),
        "quarter": compute_quarter(int(request["month"])),
        "month": int(request["month"]),
        "day_of_month": int(request["day_of_month"]),
        "day_of_week": int(request["day_of_week"]),
        "day_of_year": flight_dt.timetuple().tm_yday,
        "week_of_year": flight_dt.isocalendar().week,
        "is_weekend": 1 if int(request["day_of_week"]) in {6, 7} else 0,
        "sched_dep_minutes": sched_dep_minutes,
        "sched_arr_minutes": sched_arr_minutes,
        "sched_dep_hour": sched_dep_hour,
        "sched_arr_hour": sched_arr_hour,
        "origin_wac": int(route_info["origin_wac"]),
        "dest_wac": int(route_info["dest_wac"]),
        "scheduled_elapsed_time": scheduled_elapsed_time,
        "distance": distance,
        "distance_group": compute_distance_group(distance),
        "season": compute_season(int(request["month"])),
        "dep_time_bucket": compute_dep_time_bucket(sched_dep_hour),
        "carrier": str(request["carrier"]).strip().upper(),
        "origin": str(request["origin"]).strip().upper(),
        "dest": str(request["dest"]).strip().upper(),
    }


def input_schema() -> StructType:
    return StructType(
        [
            StructField("year", IntegerType(), False),
            StructField("quarter", IntegerType(), False),
            StructField("month", IntegerType(), False),
            StructField("day_of_month", IntegerType(), False),
            StructField("day_of_week", IntegerType(), False),
            StructField("day_of_year", IntegerType(), False),
            StructField("week_of_year", IntegerType(), False),
            StructField("is_weekend", IntegerType(), False),
            StructField("sched_dep_minutes", IntegerType(), False),
            StructField("sched_arr_minutes", IntegerType(), False),
            StructField("sched_dep_hour", IntegerType(), False),
            StructField("sched_arr_hour", IntegerType(), False),
            StructField("origin_wac", IntegerType(), False),
            StructField("dest_wac", IntegerType(), False),
            StructField("scheduled_elapsed_time", DoubleType(), False),
            StructField("distance", DoubleType(), False),
            StructField("distance_group", IntegerType(), False),
            StructField("season", StringType(), False),
            StructField("dep_time_bucket", StringType(), False),
            StructField("carrier", StringType(), False),
            StructField("origin", StringType(), False),
            StructField("dest", StringType(), False),
        ]
    )


def predict_delay(request: dict[str, Any], spark, model) -> dict[str, Any]:
    features = build_feature_row(request)
    input_df = spark.createDataFrame([features], schema=input_schema())
    prediction_row = model.transform(input_df).select("prediction", "probability").first()
    if prediction_row is None:
        raise RuntimeError("Prediction failed.")

    probability_delay_15 = float(prediction_row["probability"][1])
    prediction_delay_15 = int(prediction_row["prediction"])
    return {
        "probability_delay_15": probability_delay_15,
        "prediction_delay_15": prediction_delay_15,
        "derived_features": features,
    }


def explain_prediction(request: dict[str, Any], prediction: dict[str, Any] | None = None) -> dict[str, Any]:
    if prediction is None:
        raise ValueError("Prediction result is required for explanation.")

    features = prediction["derived_features"]
    probability = float(prediction["probability_delay_15"])
    reasons: list[str] = []

    if features["dep_time_bucket"] in {"evening", "overnight"}:
        reasons.append("The flight departs later in the day, when delay propagation is usually stronger.")
    if float(features["distance"]) >= 1500:
        reasons.append("This is a long-haul route, which usually has more operational exposure.")
    if features["season"] in {"summer", "winter"}:
        reasons.append("The travel season is historically more disruption-prone.")
    if int(features["is_weekend"]) == 1:
        reasons.append("Weekend schedules often behave differently from weekday business traffic.")
    if not reasons:
        reasons.append("The route and schedule look relatively stable in the baseline feature set.")

    band = "high" if probability >= 0.65 else "medium" if probability >= 0.35 else "low"
    return {
        "risk_band": band,
        "summary": f"Estimated delay>=15m probability is {probability:.1%}, which falls in the {band} risk band.",
        "reasons": reasons,
        "route_info": get_route_info(str(request["origin"]), str(request["dest"])),
    }
