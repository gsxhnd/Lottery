"""原始 SSQ 文本行解析"""

from collections.abc import Iterator
from pathlib import Path

from lottery.domain.types import LotteryRecord


def parse_raw_line(line: str) -> LotteryRecord | None:
    """解析单行原始数据，列不足或格式错误时返回 None。"""
    parts = line.strip().split()
    if len(parts) < 9:
        return None

    try:
        red_balls = [int(parts[i]) for i in range(2, 8)]
        blue_ball = int(parts[8])
    except ValueError:
        return None

    return LotteryRecord(
        issue=parts[0],
        date=parts[1],
        red_balls=red_balls,
        blue_ball=blue_ball,
    )


def iter_raw_records(file_path: str | Path) -> Iterator[LotteryRecord]:
    """逐行迭代原始文件中的有效开奖记录。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"原始数据文件不存在: {path}")

    with path.open(encoding="utf-8") as f:
        for line in f:
            record = parse_raw_line(line)
            if record is not None:
                yield record
