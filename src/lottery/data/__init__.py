"""数据处理模块"""

from .dataset import (
    LotteryDataset,
    build_sequence_tensor,
    denormalize_prediction,
    normalize_record,
)
from .duckdb import LotteryDataStore, SyncResult
from .loader import load_lottery_data
from .repository import get_data_store, load_lottery_records, sync_data

__all__ = [
    "LotteryDataStore",
    "LotteryDataset",
    "SyncResult",
    "build_sequence_tensor",
    "denormalize_prediction",
    "get_data_store",
    "load_lottery_data",
    "load_lottery_records",
    "normalize_record",
    "sync_data",
]
