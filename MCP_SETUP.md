# MCP Setup

This project includes a minimal MCP stdio server:

- [bts_delay_mcp_server.py](/Users/dingyuyao/Documents/SJSU/Spring2026/DATA228/GroupProject/bts_delay_mcp_server.py)

It exposes three tools:

- `predict_delay`
- `get_route_info`
- `explain_prediction`

## Run the API + UI

From the project root:

```bash
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

Open either UI:

```text
http://127.0.0.1:8000/app
http://127.0.0.1:8000/app_multimodel
```

Use:

- `/app` for the original single-model baseline page
- `/app_multimodel` for the multi-model selector page

The UI includes an MCP-style chatbox backed by the same project tools.

## Multi-Model Deployment Behavior

The backend now supports deployed model selection for prediction and chat.

Currently expected local model directories:

```text
data/model/bts_delay_lr_baseline
data/model/bts_delay_rf_best
data/model/bts_delay_gbt_best
data/model/bts_delay_best_recent_3models
```

Behavior:

- the multi-model page sends `model_id` with each prediction request
- the backend lazily loads `lr`, `rf`, and `gbt` models
- the chat endpoint uses the same selected `model_id` from the current form context
- the UI's `recommended model` label is inferred from:

```text
data/model/bts_delay_best_recent_3models
```

Optional override:

```bash
RECOMMENDED_MODEL_PATH=data/model/another_best_model_dir python -m uvicorn bts_delay_api:app --reload
```

## Enable LLM + Tool Calling

The chatbox can optionally use the OpenAI Responses API to decide which tool to call.

Set:

```bash
export OPENAI_API_KEY=your_key_here
```

Optional model override:

```bash
export OPENAI_MODEL=gpt-5-mini
```

Then start the API:

```bash
source .venv/bin/activate
python -m uvicorn bts_delay_api:app --reload
```

Behavior:

- If `OPENAI_API_KEY` is set, `/chat` uses the LLM agent over project tools.
- If `OPENAI_API_KEY` is not set, `/chat` falls back to the local rule-based router.
- In both cases, chat uses the current selected model from the multi-model form when `flight_context.model_id` is present.

## Run the MCP Server

From the project root:

```bash
source .venv/bin/activate
python bts_delay_mcp_server.py
```

The server uses stdio and implements these MCP methods:

- `initialize`
- `tools/list`
- `tools/call`

## Tool Inputs

### `predict_delay`

Required fields:

- `model_id` (optional, e.g. `lr`, `rf`, `gbt`)
- `year`
- `month`
- `day_of_month`
- `day_of_week`
- `carrier`
- `origin`
- `dest`
- `crs_dep_time`
- `crs_arr_time`

Time fields use `HHMM` integers.

### `get_route_info`

Required fields:

- `origin`
- `dest`

### `explain_prediction`

Uses the same input schema as `predict_delay`.
