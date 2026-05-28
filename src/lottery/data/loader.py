"""从原始文本文件加载数据（兼容旧接口）"""

from lottery.data.parser import iter_raw_records
from lottery_data import LotteryRecord


def load_lottery_data(file_path: str) -> list[LotteryRecord]:
    """加载双色球历史数据

    Args:
        file_path: 数据文件路径

    Returns:
        开奖记录列表
    """
    return list(iter_raw_records(file_path))
