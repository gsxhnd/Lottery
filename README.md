# Lottery

基于 PyTorch LSTM 的双色球（SSQ）历史数据实验项目。

> 项目定位是学习、实验与工程练习，不提供任何投注建议，也不承诺预测有效性。

## 当前状态

- ✅ 里程碑一：工程骨架
- ✅ 里程碑二：最小训练闭环（LSTM 训练 + TensorBoard）
- ✅ 里程碑三：最小推理闭环
- ✅ 数据层：DuckDB 管道（raw → 库、增量同步、训练读数）

## 快速开始

```bash
uv sync
mkdir -p data
curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt
uv run lottery data sync --full   # 导入 DuckDB
uv run lottery train              # 训练（默认从 DuckDB 读数据）
```

详见 [快速开始](docs/01_quickstart.md)。

## 文档

| 文档 | 内容 |
|------|------|
| [快速开始](docs/01_quickstart.md) | 环境、数据、DuckDB、训练 |
| [CLI 参考](docs/02_cli.md) | `data` / `train` / `predict` |
| [配置指南](docs/03_config.md) | `raw_file`、`db_file`、`source` 等 |
| [数据管道](docs/05_data_pipeline.md) | 解析、同步、样本构造 |
| [架构设计](docs/04_architecture.md) | 分层与数据流 |
| [模块接口](docs/07_modules.md) | API 与依赖 |
| [文档索引](docs/README.md) | 全部文档导航 |

## 仓库结构

```text
Lottery/
├── data/                    # 本地数据（gitignore）：raw_ssq.txt、lottery.duckdb
├── docs/                    # 文档
├── src/lottery/             # 源码
│   ├── cli/                 # 命令行入口（含 data 子命令）
│   ├── config/              # 配置加载
│   ├── domain/              # 领域对象
│   ├── data/                # 解析、DuckDB、Dataset
│   │   └── duckdb/          # 存储与同步
│   ├── models/              # 模型定义
│   ├── training/            # 训练流程
│   └── inference/           # 推理执行
├── config/
│   └── config.toml.example  # 配置示例
├── pyproject.toml
└── README.md
```
