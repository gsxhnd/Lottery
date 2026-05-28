"""DuckDB 读写与 raw 同步"""

from dataclasses import dataclass
from pathlib import Path

import duckdb

from lottery.data.duckdb import schema
from lottery.data.parser import iter_raw_records
from lottery.domain.types import LotteryRecord


@dataclass(frozen=True)
class SyncResult:
    """raw → DuckDB 同步结果"""

    inserted: int
    skipped: int
    total_in_db: int
    mode: str


class LotteryDataStore:
    """双色球开奖记录的 DuckDB 存储。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.db_path))

    def ensure_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(schema.CREATE_DRAWS_TABLE)
        conn.execute(schema.CREATE_DRAWS_DATE_INDEX)

    def count(self, conn: duckdb.DuckDBPyConnection | None = None) -> int:
        own_conn = conn is None
        if own_conn:
            conn = self.connect()
        try:
            self.ensure_schema(conn)
            row = conn.execute(
                f"SELECT COUNT(*) FROM {schema.DRAWS_TABLE}"
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            if own_conn:
                conn.close()

    def sync_full(self, raw_file: str | Path) -> SyncResult:
        """全量重建：清空表后从 raw 重新导入。"""
        records = list(iter_raw_records(raw_file))
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            conn.execute(f"DELETE FROM {schema.DRAWS_TABLE}")
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

    def sync_incremental(self, raw_file: str | Path) -> SyncResult:
        """增量同步：仅插入库中不存在的期号。"""
        conn = self.connect()
        try:
            self.ensure_schema(conn)
            existing = {
                row[0]
                for row in conn.execute(
                    f"SELECT issue FROM {schema.DRAWS_TABLE}"
                ).fetchall()
            }

            to_insert: list[LotteryRecord] = []
            skipped = 0
            for record in iter_raw_records(raw_file):
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
            rows = conn.execute(schema.SELECT_ALL_RECORDS).fetchall()
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
        conn.executemany(schema.INSERT_DRAW, params)
        return len(records)


def _row_to_record(row: tuple) -> LotteryRecord:
    issue, draw_date, r1, r2, r3, r4, r5, r6, blue = row
    return LotteryRecord(
        issue=str(issue),
        date=str(draw_date),
        red_balls=[int(r1), int(r2), int(r3), int(r4), int(r5), int(r6)],
        blue_ball=int(blue),
    )
