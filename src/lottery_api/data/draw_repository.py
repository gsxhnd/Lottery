"""API 数据访问层。"""

from lottery.data.repository import get_data_store
from lottery_data import LotteryRecord


class DrawRepository:
    """读取开奖数据。"""

    def __init__(self, config: dict) -> None:
        self._store = get_data_store(config)

    def fetch_records(self) -> list[LotteryRecord]:
        return self._store.fetch_records()
