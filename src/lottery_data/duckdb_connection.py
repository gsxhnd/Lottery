"""DuckDB 连接管理。"""

from pathlib import Path

import duckdb


class DuckDBConnectionFactory:
    """DuckDB 连接工厂。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.db_path))
