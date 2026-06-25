"""独立数据包：数据定义与 DuckDB 读写。"""

from .duckdb_connection import DuckDBConnectionFactory
from .duckdb_repository import LotteryDuckDBRepository
from .models import LotteryRecord, SyncResult
from .repository import get_db_path, get_repository, get_repository_for_path

__all__ = [
    "DuckDBConnectionFactory",
    "LotteryDuckDBRepository",
    "LotteryRecord",
    "SyncResult",
    "get_db_path",
    "get_repository",
    "get_repository_for_path",
]
