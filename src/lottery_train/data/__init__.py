"""数据处理模块"""

from .dataset import (
    LotteryDataset,
    build_sequence_tensor,
    denormalize_prediction,
    normalize_record,
)
from .loader import load_lottery_data
from .repository import load_lottery_records, sync_data
from lottery_data import SyncResult

__all__ = [
    "LotteryDataset",
    "SyncResult",
    "build_sequence_tensor",
    "denormalize_prediction",
    "load_lottery_data",
    "load_lottery_records",
    "normalize_record",
    "sync_data",
]
