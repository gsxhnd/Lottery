# AGENTS.md

## Toolchains

- Python uses `uv`, not `pip` or `poetry`; `.python-version` is `3.12`, setup with `uv sync`.
- Python scripts are `uv run lottery ...` (`lottery.cli:main`) and `uv run lottery-api` (`lottery_api.__main__:main`) from `pyproject.toml`.
- Frontend is a separate npm app in `web/` with `package-lock.json`; run npm commands from `web/`.

## Commands

- `uv run lottery --help` is the fastest Python smoke test.
- `uv run lottery data sync --full` imports `data/raw_ssq.txt` into `data/lottery.duckdb`; omit `--full` for incremental sync.
- `uv run lottery data status` verifies DuckDB row count and issue range.
- `uv run lottery train [--config PATH]` trains and writes `output/models/{timestamp}/` plus TensorBoard logs.
- `uv run lottery predict --model <model-dir-or-model.pt> [--config PATH]` prints JSON and writes `output/summaries/`.
- `uv run lottery-api [--config PATH] [--reload]` starts FastAPI at `127.0.0.1:8000` with `/docs` and static UI if `static/` exists.
- In `web/`: `npm run dev`, `npm run build`, `npm run lint`, `npm run typecheck`, `npm run format`.

## Architecture

- Training is wired in `src/lottery/cli/main.py:_train()`: `load_config` → `LotteryDataset.from_config(..., seq_len=10)` → `DataLoader` → `LotteryLSTM()` → `Trainer.train()` → `save_model`.
- Data access goes through `src/lottery/data/repository.py`; `source=auto` uses DuckDB when it exists and has rows, otherwise falls back to raw text.
- DuckDB implementation is in `src/lottery_data/`; older docs may still mention `src/lottery/data/duckdb/`.
- FastAPI lives in `src/lottery_api/`; `PredictionService` caches loaded records and model state, and `/data/reload` refreshes records.
- The Vite app in `web/` builds to repo-root `static/`, which `lottery-api` serves at `/` and `/assets`.

## Local Data And Config

- `data/`, `output/`, and `static/` are gitignored generated/local state; do not assume they exist in a fresh checkout.
- `data/raw_ssq.txt` is required for sync and for raw fallback; README uses `curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt`.
- Default config path is `config/config.toml`; if missing, `src/lottery/config/loader.py` falls back to hardcoded defaults merged like `config/config.toml.example`.
- `seq_len=10` in training/prediction and default `LotteryLSTM()` model dimensions are still hardcoded, not config-driven.

## Validation Notes

- There is no Python test/lint/typecheck config in this repo; use `uv run lottery --help` or a focused CLI/API run for backend smoke tests.
- Frontend verification is available: `npm run lint`, `npm run typecheck`, and `npm run build` from `web/`.
- For full-stack UI work, run `uv run lottery-api` and `npm run dev` from `web/`; Vite proxies API paths to `http://127.0.0.1:8000`.

## Source Of Truth

- Trust executable config and source first: `pyproject.toml`, `web/package.json`, `src/lottery/`, `src/lottery_api/`, `src/lottery_data/`, then docs.
- Detailed module/docs references are in `docs/07_modules.md` and `docs/08_conventions.md`, but verify stale paths against source before editing.
