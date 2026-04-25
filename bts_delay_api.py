#!/usr/bin/env python3
"""FastAPI service for BTS arrival delay >= 15 minute prediction.

Run locally:
  uvicorn bts_delay_api:app --reload

Optional environment variables:
  MODEL_PATH=data/model/bts_delay_lr_baseline
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from delay_tools import (
    explain_prediction,
    get_route_info,
    predict_delay as predict_delay_with_model,
)
from llm_agent import openai_available, run_llm_chat


MODEL_PATH = os.environ.get("MODEL_PATH", "data/model/bts_delay_lr_baseline")
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

spark: SparkSession | None = None
model: PipelineModel | None = None


class DelayPredictionRequest(BaseModel):
    year: int = Field(..., ge=1987, le=2100)
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=1, le=7, description="1=Monday, 7=Sunday")
    carrier: str = Field(..., min_length=2, max_length=10)
    origin: str = Field(..., min_length=3, max_length=3)
    dest: str = Field(..., min_length=3, max_length=3)
    crs_dep_time: int = Field(..., ge=0, le=2359, description="scheduled departure time in HHMM")
    crs_arr_time: int = Field(..., ge=0, le=2359, description="scheduled arrival time in HHMM")

    @field_validator("carrier", "origin", "dest")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("crs_dep_time", "crs_arr_time")
    @classmethod
    def validate_hhmm(cls, value: int) -> int:
        hours = value // 100
        minutes = value % 100
        if hours > 23 or minutes > 59:
            raise ValueError("time must be in HHMM format")
        return value


class DelayPredictionResponse(BaseModel):
    probability_delay_15: float
    prediction_delay_15: int
    derived_features: dict[str, int | float | str]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    flight_context: DelayPredictionRequest | None = None


class ChatResponse(BaseModel):
    assistant_message: str
    tool_name: str
    tool_output: dict[str, object]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global spark, model
    spark = SparkSession.builder.appName("BtsDelayApi").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(MODEL_PATH)
    try:
        yield
    finally:
        if spark is not None:
            spark.stop()
            spark = None


app = FastAPI(
    title="BTS Delay Prediction API",
    version="0.1.0",
    description="Predict whether a flight will arrive 15 or more minutes late.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    if spark is None or model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "BTS delay prediction API is running.",
        "app": "/app",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict_delay",
        "chat": "/chat",
    }


@app.get("/app")
def app_page() -> FileResponse:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend not found")
    return FileResponse(FRONTEND_PATH)


@app.post("/predict_delay", response_model=DelayPredictionResponse)
def predict_delay(request: DelayPredictionRequest) -> DelayPredictionResponse:
    if spark is None or model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        result = predict_delay_with_model(request.model_dump(), spark, model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DelayPredictionResponse(**result)


def infer_route_codes(message: str) -> tuple[str, str] | None:
    codes = re.findall(r"\b[A-Z]{3}\b", message.upper())
    if len(codes) >= 2:
        return codes[0], codes[1]
    return None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if spark is None or model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    message = request.message.strip()
    lowered = message.lower()
    context = request.flight_context.model_dump() if request.flight_context is not None else None

    try:
        if openai_available():
            llm_result = run_llm_chat(message, context, spark, model)
            return ChatResponse(
                assistant_message=llm_result["assistant_message"],
                tool_name=llm_result["tool_name"],
                tool_output=llm_result["tool_output"],
            )

        if any(keyword in lowered for keyword in ("route", "distance", "airport", "wac")):
            if context is not None:
                origin = context["origin"]
                dest = context["dest"]
            else:
                inferred = infer_route_codes(message)
                if inferred is None:
                    raise ValueError("Provide a flight context or mention both three-letter airport codes.")
                origin, dest = inferred

            tool_output = get_route_info(origin, dest)
            assistant_message = (
                f"Route {tool_output['origin']} to {tool_output['dest']} is about "
                f"{tool_output['distance_miles']:.1f} miles. "
                f"Origin WAC is {tool_output['origin_wac']} and destination WAC is {tool_output['dest_wac']}."
            )
            return ChatResponse(
                assistant_message=assistant_message,
                tool_name="get_route_info",
                tool_output=tool_output,
            )

        if any(keyword in lowered for keyword in ("why", "explain", "reason")):
            if context is None:
                raise ValueError("Explain requests need the current flight context from the form.")
            prediction = predict_delay_with_model(context, spark, model)
            tool_output = explain_prediction(context, prediction)
            assistant_message = tool_output["summary"] + " " + " ".join(tool_output["reasons"])
            return ChatResponse(
                assistant_message=assistant_message,
                tool_name="explain_prediction",
                tool_output=tool_output,
            )

        if any(keyword in lowered for keyword in ("predict", "delay", "risk", "late")):
            if context is None:
                raise ValueError("Prediction requests need the current flight context from the form.")
            tool_output = predict_delay_with_model(context, spark, model)
            band = "high" if tool_output["probability_delay_15"] >= 0.65 else "medium" if tool_output["probability_delay_15"] >= 0.35 else "low"
            assistant_message = (
                f"Estimated delay>=15m probability is {tool_output['probability_delay_15']:.1%}. "
                f"The current flight is in the {band} risk band."
            )
            return ChatResponse(
                assistant_message=assistant_message,
                tool_name="predict_delay",
                tool_output=tool_output,
            )

        return ChatResponse(
            assistant_message=(
                "I can predict delay risk, explain the current prediction, or report route metadata. "
                "Try: 'predict this flight', 'explain this prediction', or 'what is the route distance?'"
            ),
            tool_name="help",
            tool_output={},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
