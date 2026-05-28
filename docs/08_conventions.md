# 工程约定

## 包管理器

- 使用 `uv`，禁止 pip/poetry
- Python 版本：>= 3.12（见 `.python-version`）
- 安装依赖：`uv sync`
- 运行命令：`uv run lottery ...`、`uv run lottery-api`（预测 API + Web GUI，默认 `http://127.0.0.1:8000/`）

## 包结构

```
src/lottery_api/ # FastAPI 预测 HTTP 接口
src/lottery/
├── cli/         # argparse 命令行入口
├── config/      # TOML 配置加载与默认值
├── domain/      # dataclass 领域对象
├── data/        # raw 解析、DuckDB 同步/查询、PyTorch Dataset
│   └── duckdb/  # 表结构与 LotteryDataStore
├── models/      # LSTM 模型定义
├── training/    # 训练循环 + 模型保存
└── inference/   # 推理执行
```

## 输出目录

```
output/
├── models/{timestamp}/
│   ├── model.pt
│   └── metadata.json
├── logs/{timestamp}/      # TensorBoard
└── summaries/             # 推理结果 JSON
```

时间戳格式：`YYYYMMDD_HHMMSS`

## 数据文件

`data/` 目录被 gitignore，本地需自备：

| 文件 | 说明 |
|------|------|
| `data/raw_ssq.txt` | 原始 SSQ 文本（[17500 数据源](https://data.17500.cn/ssq_asc.txt)） |
| `data/lottery.duckdb` | 由 `uv run lottery data sync` 生成，训练默认读此库 |

## 验证

- 冒烟测试：`uv run lottery --help`
- 数据同步：`uv run lottery data sync --full` → `uv run lottery data status`
- 训练测试：`uv run lottery train`
- TensorBoard：`tensorboard --logdir=output/logs`

---

## 已知技术债

- `seq_len=10` 硬编码在 `cli/main.py:64`，未从配置读取
- `LotteryLSTM()` 构造参数硬编码，未从配置读取
- `predict` 写入 `output/summaries/{timestamp}_prediction.json`
- 无测试、linter、formatter、类型检查
- `.ruff_cache/` 存在但无 ruff 配置

### 未接入配置的硬编码参数

| 参数 | 硬编码位置 | 当前值 |
|------|-----------|--------|
| `seq_len` | `cli/main.py:64` | 10 |
| `input_size` | `LotteryLSTM()` 默认参数 | 7 |
| `hidden_size` | `LotteryLSTM()` 默认参数 | 64 |
| `num_layers` | `LotteryLSTM()` 默认参数 | 2 |
