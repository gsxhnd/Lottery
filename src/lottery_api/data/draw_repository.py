"""API 数据访问层。"""

from lottery_data import LotteryRecord, get_repository


class DrawRepository:
    """读取开奖数据。"""

    def __init__(self, config: dict) -> None:
        self._store = get_repository(config)

    def fetch_records(self) -> list[LotteryRecord]:
        return self._store.fetch_records()
