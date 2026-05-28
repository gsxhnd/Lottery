# AGENTS.md

## Workflow

- Use `uv`, not `pip` or `poetry`.
- Python 3.12 (`.python-version`); install deps with `uv sync`.
- CLI entrypoint: `uv run lottery ...` (wired via `pyproject.toml` `[project.scripts]` → `lottery.cli:main`).
- Full engineering conventions: see `docs/08_conventions.md`.

## Verified Commands

- `uv run lottery --help` — show CLI help.
- `uv run lottery train` — run training (requires `data/raw_ssq.txt`).
- `uv run lottery train --config config/config.toml` — train with custom config.
- `uv run lottery predict --model <path>` — load saved model and predict next draw (JSON + `output/summaries/`).
- `uv run lottery-api` — start FastAPI prediction server (Web GUI at `/`, OpenAPI at `/docs`, default `http://127.0.0.1:8000`).
- `tensorboard --logdir=output/logs` — view training logs.

## Architecture

Training flow is wired in `src/lottery/cli/main.py:_train()`:
`load_config` → `load_lottery_data` → `LotteryDataset` → `DataLoader` → `LotteryLSTM` → `Trainer.train()` → `save_model`.

Package layout, output directories, and data format: see `docs/08_conventions.md`.
Module API signatures and dependency graph: see `docs/07_modules.md`.

## Important Constraints

- `data/` is gitignored. Training requires a local `data/raw_ssq.txt` (space-separated SSQ records from https://data.17500.cn/ssq_asc.txt or https://data.17500.cn/ssq_desc.txt). Missing file → `FileNotFoundError`.
- `config/config.toml` is optional. When absent, `config/loader.py` falls back to hardcoded defaults. Copy `config/config.toml.example` for reproducible settings.
- Hardcoded parameters not yet in config: see `docs/08_conventions.md` "已知技术债" section.
- PyTorch source varies by platform: CUDA 13.0 index on Windows, CPU index on Linux, default PyPI on macOS.

## Validation

- No test suite, linter, formatter, or type checker is configured.
- Smoke-test changes with `uv run lottery --help` or a training run.

## Docs Trust Order

When docs conflict with code, trust in this order:
1. `src/lottery/` source code and `pyproject.toml`
2. `AGENTS.md` (this file)
3. `docs/` documentation
