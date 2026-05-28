"""训练/推理统一的数据访问入口"""

from pathlib import Path
from typing import Any

from lottery.data.duckdb import LotteryDataStore, SyncResult
from lottery.data.loader import load_lottery_data
from lottery.domain.types import LotteryRecord


def get_db_path(config: dict[str, Any]) -> str:
    return config["data"].get("db_file", "data/lottery.duckdb")


def get_data_store(config: dict[str, Any]) -> LotteryDataStore:
    return LotteryDataStore(get_db_path(config))


def sync_data(
    config: dict[str, Any],
    *,
    full: bool = False,
) -> SyncResult:
    """将 raw 文件同步到 DuckDB。"""
    raw_file = config["data"]["raw_file"]
    store = get_data_store(config)
    if full:
        return store.sync_full(raw_file)
    return store.sync_incremental(raw_file)


def load_lottery_records(config: dict[str, Any]) -> list[LotteryRecord]:
    """按配置加载开奖记录，供训练与推理使用。

    source 取值：
    - ``raw``：始终读原始文本
    - ``duckdb``：仅从 DuckDB 读（库为空则报错）
    - ``auto``（默认）：DuckDB 存在且有条目时用库，否则回退 raw
    """
    data_cfg = config["data"]
    raw_file = data_cfg["raw_file"]
    source = data_cfg.get("source", "auto")
    db_path = Path(get_db_path(config))

    if source == "raw":
        return load_lottery_data(raw_file)

    if source in ("duckdb", "auto") and db_path.exists():
        store = LotteryDataStore(db_path)
        records = store.fetch_records()
        if records:
            return records
        if source == "duckdb":
            raise RuntimeError(
                f"DuckDB 中无开奖记录，请先执行: uv run lottery data sync ({db_path})"
            )

    return load_lottery_data(raw_file)
