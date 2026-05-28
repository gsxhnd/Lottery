"""独立数据包：数据定义、DuckDB 连接与请求。"""

from .duckdb_connection import DuckDBConnectionFactory
from .duckdb_repository import LotteryDuckDBRepository
from .models import LotteryRecord, SyncResult

__all__ = [
    "DuckDBConnectionFactory",
    "LotteryDuckDBRepository",
    "LotteryRecord",
    "SyncResult",
]
