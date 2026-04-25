#!/usr/bin/env python3
"""Minimal MCP stdio server for the BTS delay project.

Implements a small subset of the MCP JSON-RPC protocol:
- initialize
- tools/list
- tools/call
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from delay_tools import explain_prediction, get_route_info, predict_delay


MODEL_PATH = os.environ.get("MODEL_PATH", "data/model/bts_delay_lr_baseline")

spark: SparkSession | None = None
model: PipelineModel | None = None


def get_runtime() -> tuple[SparkSession, PipelineModel]:
    global spark, model
    if spark is None:
        spark = SparkSession.builder.appName("BtsDelayMcpServer").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    if model is None:
        model = PipelineModel.load(MODEL_PATH)
    return spark, model


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("utf-8").split(":", 1)
        headers[name.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None
    body = sys.stdin.buffer.read(content_length)
    return json.loads(body.decode("utf-8"))


def write_message(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "predict_delay",
            "description": "Predict whether a scheduled flight will arrive 15+ minutes late.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "month": {"type": "integer"},
                    "day_of_month": {"type": "integer"},
                    "day_of_week": {"type": "integer"},
                    "carrier": {"type": "string"},
                    "origin": {"type": "string"},
                    "dest": {"type": "string"},
                    "crs_dep_time": {"type": "integer", "description": "HHMM"},
                    "crs_arr_time": {"type": "integer", "description": "HHMM"},
                },
                "required": [
                    "year",
                    "month",
                    "day_of_month",
                    "day_of_week",
                    "carrier",
                    "origin",
                    "dest",
                    "crs_dep_time",
                    "crs_arr_time",
                ],
            },
        },
        {
            "name": "get_route_info",
            "description": "Return route metadata such as WAC codes and approximate route distance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "dest": {"type": "string"},
                },
                "required": ["origin", "dest"],
            },
        },
        {
            "name": "explain_prediction",
            "description": "Explain the current delay-risk prediction using the baseline feature set.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "month": {"type": "integer"},
                    "day_of_month": {"type": "integer"},
                    "day_of_week": {"type": "integer"},
                    "carrier": {"type": "string"},
                    "origin": {"type": "string"},
                    "dest": {"type": "string"},
                    "crs_dep_time": {"type": "integer", "description": "HHMM"},
                    "crs_arr_time": {"type": "integer", "description": "HHMM"},
                },
                "required": [
                    "year",
                    "month",
                    "day_of_month",
                    "day_of_week",
                    "carrier",
                    "origin",
                    "dest",
                    "crs_dep_time",
                    "crs_arr_time",
                ],
            },
        },
    ]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "get_route_info":
        return get_route_info(arguments["origin"], arguments["dest"])

    runtime_spark, runtime_model = get_runtime()
    if name == "predict_delay":
        return predict_delay(arguments, runtime_spark, runtime_model)

    if name == "explain_prediction":
        prediction = predict_delay(arguments, runtime_spark, runtime_model)
        return explain_prediction(arguments, prediction)

    raise ValueError(f"Unknown tool: {name}")


def success_response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def error_response(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    message_id = message.get("id")

    if message_id is None:
        return None

    try:
        if method == "initialize":
            return success_response(
                message_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "bts-delay-mcp", "version": "0.1.0"},
                },
            )

        if method == "tools/list":
            return success_response(message_id, {"tools": tool_definitions()})

        if method == "tools/call":
            params = message.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            tool_result = call_tool(tool_name, arguments)
            return success_response(
                message_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_result, indent=2),
                        }
                    ]
                },
            )

        return error_response(message_id, -32601, f"Method not found: {method}")
    except Exception as exc:  # noqa: BLE001
        return error_response(message_id, -32000, str(exc))


def main() -> int:
    try:
        while True:
            message = read_message()
            if message is None:
                break
            response = handle_message(message)
            if response is not None:
                write_message(response)
    finally:
        if spark is not None:
            spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
