"""数据加载器"""

from pathlib import Path
from lottery.domain.types import LotteryRecord


def load_lottery_data(file_path: str) -> list[LotteryRecord]:
    """加载双色球历史数据

    Args:
        file_path: 数据文件路径

    Returns:
        开奖记录列表
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            issue = parts[0]
            date = parts[1]
            red_balls = [int(parts[i]) for i in range(2, 8)]
            blue_ball = int(parts[8])

            records.append(
                LotteryRecord(
                    issue=issue, date=date, red_balls=red_balls, blue_ball=blue_ball
                )
            )

    return records
