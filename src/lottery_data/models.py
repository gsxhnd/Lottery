"""lottery 数据模型定义。"""

from dataclasses import dataclass


@dataclass
class LotteryRecord:
    """单期开奖记录。"""

    issue: str
    date: str
    red_balls: list[int]
    blue_ball: int


@dataclass(frozen=True)
class SyncResult:
    """raw -> DuckDB 同步结果。"""

    inserted: int
    skipped: int
    total_in_db: int
    mode: str
