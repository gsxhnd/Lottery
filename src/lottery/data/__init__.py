"""数据处理模块"""

from .loader import load_lottery_data
from .dataset import LotteryDataset

__all__ = ["load_lottery_data", "LotteryDataset"]
