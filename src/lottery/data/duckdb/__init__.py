"""DuckDB 持久化与同步"""

from .store import LotteryDataStore, SyncResult

__all__ = ["LotteryDataStore", "SyncResult"]
