#!/usr/bin/env python3
"""FastAPI service for BTS arrival delay >= 15 minute prediction.

Run locally:
  uvicorn bts_delay_api:app --reload

Optional environment variables:
  MODEL_PATH=data/model/bts_delay_lr_baseline
  DEFAULT_MODEL_ID=lr
  RECOMMENDED_MODEL_PATH=data/model/bts_delay_best_recent_3models
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "lr").strip().lower()
RECOMMENDED_MODEL_PATH = os.environ.get(
    "RECOMMENDED_MODEL_PATH",
    "data/model/bts_delay_best_recent_3models",
)
BASE_DIR = Path(__file__).resolve().parent
SINGLE_MODEL_FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
MULTI_MODEL_FRONTEND_PATH = BASE_DIR / "frontend" / "index_multimodel.html"

spark: SparkSession | None = None
loaded_models: dict[str, PipelineModel] = {}


def humanize_model_family(stage_name: str) -> str:
    mapping = {
        "LogisticRegressionModel": "Logistic Regression",
        "RandomForestClassificationModel": "Random Forest",
        "GBTClassificationModel": "Gradient-Boosted Trees",
    }
    return mapping.get(stage_name, stage_name.replace("Model", "").replace("Classification", " Classification"))


def short_model_family(model_family: str) -> str:
    mapping = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "Gradient-Boosted Trees": "GBT",
    }
    return mapping.get(model_family, model_family)


def infer_model_metadata(
    model_id: str,
    model_path: str,
    loaded_model: PipelineModel | None,
) -> dict[str, str]:
    model_dir = Path(model_path).name
    model_family = "Unknown"
    if loaded_model is not None and loaded_model.stages:
        model_family = humanize_model_family(type(loaded_model.stages[-1]).__name__)
    model_short = short_model_family(model_family)

    source_label = "Deployed Model"
    source_detail = f"{model_short} Active"
    lower_path = model_path.lower()
    if "best" in lower_path:
        source_label = "Selected Model"
        source_detail = f"{model_short} Best"
    elif "baseline" in lower_path:
        source_label = "Baseline Model"
        source_detail = f"{model_short} Baseline"

    return {
        "model_id": model_id,
        "model_family": model_family,
        "model_path": model_path,
        "model_dir": model_dir,
        "model_short": model_short,
        "source_label": source_label,
        "source_detail": source_detail,
        "feature_layer": "Shared feature set",
        "prediction_endpoint": "/predict_delay",
    }


def build_model_registry() -> dict[str, dict[str, str]]:
    registry = {
        "lr": {"path": "data/model/bts_delay_lr_baseline", "label": "LR Baseline"},
        "rf": {"path": "data/model/bts_delay_rf_best", "label": "RF Best"},
        "gbt": {"path": "data/model/bts_delay_gbt_best", "label": "GBT Best"},
    }
    for model_id, entry in registry.items():
        if model_id == "lr":
            entry["path"] = os.environ.get("MODEL_PATH_LR", entry["path"])
        elif model_id == "rf":
            entry["path"] = os.environ.get("MODEL_PATH_RF", entry["path"])
        elif model_id == "gbt":
            entry["path"] = os.environ.get("MODEL_PATH_GBT", entry["path"])
    return registry


MODEL_REGISTRY = build_model_registry()

MODEL_ID_BY_FAMILY = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "Gradient-Boosted Trees": "gbt",
}


def infer_deployed_model_id() -> str:
    for model_id, entry in MODEL_REGISTRY.items():
        if Path(entry["path"]).as_posix() == Path(MODEL_PATH).as_posix():
            return model_id
    return DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in MODEL_REGISTRY else "lr"


DEPLOYED_MODEL_ID = infer_deployed_model_id()


def infer_model_id_from_path(model_path: str) -> str | None:
    normalized_path = Path(model_path).as_posix()
    for model_id, entry in MODEL_REGISTRY.items():
        if Path(entry["path"]).as_posix() == normalized_path:
            return model_id

    path_obj = Path(model_path)
    if not path_obj.exists():
        return None

    inferred_model = PipelineModel.load(model_path)
    if not inferred_model.stages:
        return None

    family = humanize_model_family(type(inferred_model.stages[-1]).__name__)
    return MODEL_ID_BY_FAMILY.get(family)


RECOMMENDED_MODEL_ID = infer_model_id_from_path(RECOMMENDED_MODEL_PATH) or DEPLOYED_MODEL_ID


def get_model_entry(model_id: str | None) -> tuple[str, dict[str, str]]:
    resolved_id = (model_id or DEPLOYED_MODEL_ID).strip().lower()
    entry = MODEL_REGISTRY.get(resolved_id)
    if entry is None:
        raise ValueError(f"Unsupported model_id: {resolved_id}")
    return resolved_id, entry


def get_model(model_id: str | None) -> tuple[str, dict[str, str], PipelineModel]:
    resolved_id, entry = get_model_entry(model_id)
    if resolved_id not in loaded_models:
        loaded_models[resolved_id] = PipelineModel.load(entry["path"])
    return resolved_id, entry, loaded_models[resolved_id]


def list_available_models() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for model_id, entry in MODEL_REGISTRY.items():
        model_obj = loaded_models.get(model_id)
        metadata = infer_model_metadata(model_id, entry["path"], model_obj)
        items.append(
            {
                "id": model_id,
                "label": entry["label"],
                "path": entry["path"],
                "model_family": metadata["model_family"],
                "source_detail": metadata["source_detail"],
            }
        )
    return items


class DelayPredictionRequest(BaseModel):
    model_id: str | None = Field(
        default=None,
        description="optional deployed model id, e.g. lr, rf, gbt",
    )
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

    @field_validator("model_id")
    @classmethod
    def normalize_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().lower() or None

    @field_validator("crs_dep_time", "crs_arr_time")
    @classmethod
    def validate_hhmm(cls, value: int) -> int:
        hours = value // 100
        minutes = value % 100
        if hours > 23 or minutes > 59:
            raise ValueError("time must be in HHMM format")
        return value


class DelayPredictionResponse(BaseModel):
    model_id: str
    model_label: str
    model_family: str
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
    global spark
    spark = SparkSession.builder.appName("BtsDelayApi").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    get_model(DEPLOYED_MODEL_ID)
    try:
        yield
    finally:
        loaded_models.clear()
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
    if spark is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    model_id, _, _ = get_model(DEPLOYED_MODEL_ID)
    return {"status": "ok", "default_model_id": model_id}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "BTS delay prediction API is running.",
        "app": "/app",
        "app_multimodel": "/app_multimodel",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict_delay",
        "chat": "/chat",
    }


@app.get("/app_config")
def app_config() -> dict[str, object]:
    model_id, entry, model = get_model(DEPLOYED_MODEL_ID)
    config = infer_model_metadata(model_id, entry["path"], model)
    config["default_model_id"] = model_id
    config["recommended_model_id"] = RECOMMENDED_MODEL_ID
    config["recommended_model_label"] = MODEL_REGISTRY[RECOMMENDED_MODEL_ID]["label"]
    config["available_models"] = list_available_models()
    return config


@app.get("/app")
def app_page() -> HTMLResponse:
    if not SINGLE_MODEL_FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend not found")
    return HTMLResponse(
        SINGLE_MODEL_FRONTEND_PATH.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/app_multimodel")
def app_multimodel_page() -> HTMLResponse:
    if not MULTI_MODEL_FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="multimodel frontend not found")
    return HTMLResponse(
        MULTI_MODEL_FRONTEND_PATH.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/predict_delay", response_model=DelayPredictionResponse)
def predict_delay(request: DelayPredictionRequest) -> DelayPredictionResponse:
    if spark is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        model_id, entry, selected_model = get_model(request.model_id)
        result = predict_delay_with_model(request.model_dump(exclude_none=True), spark, selected_model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    metadata = infer_model_metadata(model_id, entry["path"], selected_model)
    return DelayPredictionResponse(
        model_id=model_id,
        model_label=entry["label"],
        model_family=metadata["model_family"],
        **result,
    )


def infer_route_codes(message: str) -> tuple[str, str] | None:
    codes = re.findall(r"\b[A-Z]{3}\b", message.upper())
    if len(codes) >= 2:
        return codes[0], codes[1]
    return None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if spark is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    message = request.message.strip()
    lowered = message.lower()
    context = request.flight_context.model_dump() if request.flight_context is not None else None
    selected_model_id = request.flight_context.model_id if request.flight_context is not None else None

    try:
        model_id, _, selected_model = get_model(selected_model_id)
        if openai_available():
            llm_result = run_llm_chat(message, context, spark, selected_model)
            llm_result["tool_output"] = {
                **llm_result["tool_output"],
                "_model_id": model_id,
            }
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
            prediction = predict_delay_with_model(context, spark, selected_model)
            tool_output = explain_prediction(context, prediction)
            tool_output["_model_id"] = model_id
            assistant_message = tool_output["summary"] + " " + " ".join(tool_output["reasons"])
            return ChatResponse(
                assistant_message=assistant_message,
                tool_name="explain_prediction",
                tool_output=tool_output,
            )

        if any(keyword in lowered for keyword in ("predict", "delay", "risk", "late")):
            if context is None:
                raise ValueError("Prediction requests need the current flight context from the form.")
            tool_output = predict_delay_with_model(context, spark, selected_model)
            tool_output["_model_id"] = model_id
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
