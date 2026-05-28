"""DuckDB 开奖数据读写。"""

from pathlib import Path

import duckdb

from . import duckdb_queries as queries
from .duckdb_connection import DuckDBConnectionFactory
from .models import LotteryRecord, SyncResult


class LotteryDuckDBRepository:
    """双色球开奖记录的 DuckDB 仓储。"""

    def __init__(self, db_path: str | Path) -> None:
        self._conn_factory = DuckDBConnectionFactory(db_path)

    def connect(self) -> duckdb.DuckDBPyConnection:
        return self._conn_factory.connect()

    def ensure_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(queries.CREATE_DRAWS_TABLE)
        conn.execute(queries.CREATE_DRAWS_DATE_INDEX)

    def count(self, conn: duckdb.DuckDBPyConnection | None = None) -> int:
        own_conn = conn is None
        if own_conn:
            conn = self.connect()
        try:
            self.ensure_schema(conn)
            row = conn.execute(queries.COUNT_DRAWS).fetchone()
            return int(row[0]) if row else 0
        finally:
            if own_conn:
                conn.close()

    def sync_full(self, records: list[LotteryRecord]) -> SyncResult:
        """全量重建：清空表后重新导入。"""
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            conn.execute(queries.DELETE_ALL_DRAWS)
            inserted = self._insert_records(conn, records)
            total = self.count(conn)
            return SyncResult(
                inserted=inserted,
                skipped=0,
                total_in_db=total,
                mode="full",
            )
        finally:
            conn.close()

    def sync_incremental(self, records: list[LotteryRecord]) -> SyncResult:
        """增量同步：仅插入库中不存在的期号。"""
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            existing = {
                row[0] for row in conn.execute(queries.SELECT_EXISTING_ISSUES).fetchall()
            }

            to_insert: list[LotteryRecord] = []
            skipped = 0
            for record in records:
                if record.issue in existing:
                    skipped += 1
                    continue
                to_insert.append(record)

            inserted = self._insert_records(conn, to_insert)
            total = self.count(conn)
            return SyncResult(
                inserted=inserted,
                skipped=skipped,
                total_in_db=total,
                mode="incremental",
            )
        finally:
            conn.close()

    def fetch_records(self) -> list[LotteryRecord]:
        """按期号升序读取全部开奖记录。"""
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            rows = conn.execute(queries.SELECT_ALL_RECORDS).fetchall()
            return [_row_to_record(row) for row in rows]
        finally:
            conn.close()

    def _insert_records(
        self,
        conn: duckdb.DuckDBPyConnection,
        records: list[LotteryRecord],
    ) -> int:
        if not records:
            return 0

        params = [
            (
                record.issue,
                record.date,
                *record.red_balls,
                record.blue_ball,
            )
            for record in records
        ]
        conn.executemany(queries.INSERT_DRAW, params)
        return len(records)


def _row_to_record(row: tuple) -> LotteryRecord:
    issue, draw_date, r1, r2, r3, r4, r5, r6, blue = row
    return LotteryRecord(
        issue=str(issue),
        date=str(draw_date),
        red_balls=[int(r1), int(r2), int(r3), int(r4), int(r5), int(r6)],
        blue_ball=int(blue),
    )
