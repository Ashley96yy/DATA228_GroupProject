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

Open:

```text
http://127.0.0.1:8000/app
```

The UI includes an MCP-style chatbox backed by the same project tools.

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
