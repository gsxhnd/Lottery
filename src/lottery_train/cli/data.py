"""data 子命令：DuckDB 同步与状态"""

import argparse
from typing import Any

from lottery_data import get_repository
from lottery_train.config import load_config
from lottery_train.data.repository import sync_data


def register_data_commands(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser("data", help="数据管道（DuckDB）")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    sync_parser = data_sub.add_parser("sync", help="将 raw 同步到 DuckDB")
    sync_parser.add_argument("--config", type=str, help="配置文件路径")
    sync_parser.add_argument(
        "--full",
        action="store_true",
        help="全量重建（清空后重导）；默认仅增量插入新期号",
    )
    sync_parser.set_defaults(handler=_sync)

    status_parser = data_sub.add_parser("status", help="查看 DuckDB 数据状态")
    status_parser.add_argument("--config", type=str, help="配置文件路径")
    status_parser.set_defaults(handler=_status)


def run_data_command(args: argparse.Namespace) -> int:
    handler = getattr(args, "handler", None)
    if handler is None:
        return 1
    return handler(args)


def _sync(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    raw_file = config["data"]["raw_file"]
    db_file = config["data"].get("db_file", "data/lottery.duckdb")
    mode = "全量" if args.full else "增量"

    print(f"同步模式: {mode}")
    print(f"原始文件: {raw_file}")
    print(f"DuckDB:   {db_file}")

    result = sync_data(config, full=args.full)
    print(
        f"完成 [{result.mode}]: 新增 {result.inserted} 条, "
        f"跳过 {result.skipped} 条, 库内共 {result.total_in_db} 条"
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = get_repository(config)
    db_file = config["data"].get("db_file", "data/lottery.duckdb")
    raw_file = config["data"]["raw_file"]
    count = store.count()

    print(f"原始文件: {raw_file}")
    print(f"DuckDB:   {db_file}")
    print(f"记录数:   {count}")

    if count > 0:
        records = store.fetch_records()
        print(f"首期:     {records[0].issue} ({records[0].date})")
        print(f"末期:     {records[-1].issue} ({records[-1].date})")

    return 0
