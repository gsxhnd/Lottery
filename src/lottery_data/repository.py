"""DuckDB 仓储工厂（按配置解析库路径）。"""

from pathlib import Path
from typing import Any

from .duckdb_repository import LotteryDuckDBRepository


def get_db_path(config: dict[str, Any]) -> str:
    return config["data"].get("db_file", "data/lottery.duckdb")


def get_repository(config: dict[str, Any]) -> LotteryDuckDBRepository:
    return LotteryDuckDBRepository(get_db_path(config))


def get_repository_for_path(db_path: str | Path) -> LotteryDuckDBRepository:
    return LotteryDuckDBRepository(db_path)
