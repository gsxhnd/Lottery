"""配置加载器"""

import tomllib
from pathlib import Path
from typing import Any


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """加载配置文件

    Args:
        config_path: 配置文件路径，默认为 config/config.toml

    Returns:
        配置字典
    """
    if config_path is None:
        config_path = "config/config.toml"

    path = Path(config_path)
    if not path.exists():
        return _default_config()

    with open(path, "rb") as f:
        config = tomllib.load(f)

    return _merge_with_defaults(config)


def _default_config() -> dict[str, Any]:
    """默认配置"""
    return {
        "data": {
            "raw_file": "data/raw_ssq.txt",
            "db_file": "data/lottery.duckdb",
            "source": "auto",
        },
        "output": {
            "base_dir": "output",
            "models_dir": "output/models",
            "logs_dir": "output/logs",
            "summaries_dir": "output/summaries",
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "val_split": 0.2,
        },
    }


def _merge_with_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """合并用户配置与默认配置"""
    defaults = _default_config()

    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            config[key] = {**value, **config[key]}

    return config
