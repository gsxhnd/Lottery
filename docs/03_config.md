# 配置指南

## 配置文件

项目使用 TOML 格式配置文件，默认路径为 `config/config.toml`。

如果文件不存在，系统将使用内置默认值。建议从示例复制：

```bash
cp config/config.toml.example config/config.toml
```

---

## 完整配置项

```toml
[data]
raw_file = "data/raw_ssq.txt"
db_file = "data/lottery.duckdb"
# source: raw | duckdb | auto
source = "auto"

[output]
base_dir = "output"
models_dir = "output/models"
logs_dir = "output/logs"
summaries_dir = "output/summaries"

[training]
epochs = 100
batch_size = 32
learning_rate = 0.001
val_split = 0.2
```

---

## 配置项说明

### `[data]` — 数据配置

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `raw_file` | string | `data/raw_ssq.txt` | 原始 SSQ 文本路径（同步与 `source=raw` 时使用） |
| `db_file` | string | `data/lottery.duckdb` | DuckDB 数据库文件路径 |
| `source` | string | `auto` | 训练/推理读数来源，见下表 |

`source` 取值：

| 值 | 说明 |
|----|------|
| `auto` | DuckDB 存在且有条目时用库，否则读 `raw_file` |
| `duckdb` | 仅读 DuckDB；库为空时报错 |
| `raw` | 始终读原始文本，不查库 |

数据同步不依赖 `source`，始终从 `raw_file` 写入 `db_file`。详见 [数据管道](./05_data_pipeline.md)。

### `[output]` — 输出配置

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `base_dir` | string | `output` | 输出根目录 |
| `models_dir` | string | `output/models` | 模型保存目录 |
| `logs_dir` | string | `output/logs` | TensorBoard 日志 |
| `summaries_dir` | string | `output/summaries` | 推理结果 JSON |

### `[training]` — 训练配置

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `epochs` | int | 100 | 训练轮数 |
| `batch_size` | int | 32 | 批次大小 |
| `learning_rate` | float | 0.001 | Adam 学习率 |
| `val_split` | float | 0.2 | 验证集比例（0 或过大时跳过划分） |

---

## 配置合并规则

用户配置与内置默认值**深度合并**：

- 用户未指定的键 → 使用默认值
- 用户指定的键 → 覆盖默认值
- 嵌套字典按 key 级别合并

例如，只写 `[training]` 节时，`data` 和 `output` 仍使用默认值（含 `db_file`、`source`）。

---

## 未从配置读取的参数

部分参数当前硬编码在代码中，尚未接入配置系统。详见 [工程约定 — 已知技术债](./08_conventions.md#已知技术债)。
