#!/usr/bin/env python3
"""OpenAI Responses API agent over project tools."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from delay_tools import explain_prediction, get_route_info, predict_delay


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_MODEL = "gpt-5-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"


def openai_available() -> bool:
    return bool(os.environ.get(OPENAI_API_KEY_ENV))


def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "predict_delay",
            "description": "Predict whether a scheduled flight will arrive 15 or more minutes late.",
            "parameters": {
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
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_route_info",
            "description": "Return route distance and airport WAC metadata for a route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "dest": {"type": "string"},
                },
                "required": ["origin", "dest"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "explain_prediction",
            "description": "Explain the current flight's delay-risk prediction using baseline features.",
            "parameters": {
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
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def call_openai(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{OPENAI_API_KEY_ENV} is not set.")

    req = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    spark,
    model,
) -> dict[str, Any]:
    if tool_name == "predict_delay":
        return predict_delay(arguments, spark, model)
    if tool_name == "get_route_info":
        return get_route_info(arguments["origin"], arguments["dest"])
    if tool_name == "explain_prediction":
        prediction = predict_delay(arguments, spark, model)
        return explain_prediction(arguments, prediction)
    raise ValueError(f"Unknown tool: {tool_name}")


def extract_output_text(response: dict[str, Any]) -> str:
    direct_text = response.get("output_text", "")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    parts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            content_type = content.get("type")
            if content_type in {"output_text", "text"}:
                text_value = content.get("text", "")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
    return "\n".join(parts).strip()


def run_llm_chat(
    message: str,
    flight_context: dict[str, Any] | None,
    spark,
    model,
) -> dict[str, Any]:
    model_name = os.environ.get(OPENAI_MODEL_ENV, DEFAULT_MODEL)
    tools = build_tools()

    context_text = (
        f"Current flight form context:\n{json.dumps(flight_context, indent=2)}"
        if flight_context is not None
        else "No current flight form context was provided."
    )

    instructions = (
        "You are the BTS Flight Delay project assistant. "
        "Use tools whenever the user asks for a delay prediction, route metadata, or an explanation. "
        "If the user refers to 'this flight', use the provided flight context. "
        "Keep responses concise and factual."
    )

    input_items: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"{context_text}\n\nUser request: {message}"},
    ]

    last_tool_name = "llm"
    last_tool_output: dict[str, Any] = {}
    while True:
        response = call_openai(
            {
                "model": model_name,
                "tools": tools,
                "input": input_items,
            }
        )

        output_items = response.get("output", [])
        function_calls = [item for item in output_items if item.get("type") == "function_call"]
        if not function_calls:
            assistant_text = extract_output_text(response)
            return {
                "assistant_message": assistant_text or "No assistant response returned.",
                "tool_name": last_tool_name,
                "tool_output": last_tool_output,
                "model": model_name,
            }

        input_items.extend(output_items)

        tool_output_payload: dict[str, Any] = {}
        for tool_call in function_calls:
            last_tool_name = tool_call["name"]
            arguments = json.loads(tool_call.get("arguments", "{}"))
            if flight_context is not None and last_tool_name in {"predict_delay", "explain_prediction"}:
                merged_arguments = {**flight_context, **arguments}
            else:
                merged_arguments = arguments
            tool_output_payload = execute_tool(last_tool_name, merged_arguments, spark, model)
            last_tool_output = tool_output_payload
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": json.dumps(tool_output_payload),
                }
            )
