# 模块接口

## 模块依赖图

```
cli/main.py
 ├── config/loader.py        → load_config()
 ├── cli/data.py             → data sync / status
 ├── data/repository.py      → load_lottery_records(), sync_data()
 ├── data/dataset.py         → LotteryDataset, LotteryDataset.from_config()
 ├── models/lstm.py          → LotteryLSTM
 └── training/
      ├── trainer.py         → Trainer
      └── saver.py           → save_model()

data/
 ├── parser.py               → parse_raw_line(), iter_raw_records()
 ├── loader.py               → load_lottery_data()  (raw 文件)
 ├── duckdb/store.py         → LotteryDataStore
 └── repository.py           → 统一读数与同步入口
```

---

## `config/loader.py`

### `load_config(config_path: str | None) → dict`

- 读取 TOML 配置文件
- 若文件不存在，回退到 `_default_config()`
- 用户配置与默认值深度合并

### 默认配置结构

```python
{
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
```

---

## `data/parser.py`

### `parse_raw_line(line: str) → LotteryRecord | None`

- 解析单行；列不足或数字非法时返回 `None`

### `iter_raw_records(file_path) → Iterator[LotteryRecord]`

- 逐行迭代有效记录；文件不存在时 `FileNotFoundError`

---

## `data/loader.py`

### `load_lottery_data(file_path: str) → list[LotteryRecord]`

- 从原始文本加载全部记录（兼容接口）

---

## `data/duckdb/store.py`

### `LotteryDataStore(db_path)`

| 方法 | 说明 |
|------|------|
| `sync_full(raw_file)` | 清空表后全量导入，返回 `SyncResult` |
| `sync_incremental(raw_file)` | 仅插入新 `issue` |
| `fetch_records()` | 按期号升序返回 `list[LotteryRecord]` |
| `count()` | 库内条数 |

### `SyncResult`

```python
@dataclass
class SyncResult:
    inserted: int
    skipped: int
    total_in_db: int
    mode: str  # "full" | "incremental"
```

---

## `data/repository.py`

### `sync_data(config, *, full=False) → SyncResult`

- 从 `config["data"]["raw_file"]` 同步到 `config["data"]["db_file"]`

### `load_lottery_records(config) → list[LotteryRecord]`

- 按 `data.source` 从 DuckDB 或 raw 加载（见 [配置指南](./03_config.md)）

### `get_data_store(config) → LotteryDataStore`

- 根据配置构造存储实例

---

## `data/dataset.py`

### `LotteryDataset(records, seq_len=10)`

- `__len__()` → `len(records) - seq_len`
- `__getitem__(idx)` → `(features, target)`
  - `features`: `(seq_len, 7)` 归一化序列
  - `target`: `(7,)` 归一化下一期

### `LotteryDataset.from_config(config, seq_len=10)`

- 调用 `load_lottery_records(config)` 后构造 Dataset

---

## `domain/types.py`

```python
@dataclass
class LotteryRecord:
    issue: str
    date: str
    red_balls: list[int]
    blue_ball: int

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float

@dataclass
class ModelArtifact:
    model_path: str
    metadata: dict

@dataclass
class PredictionResult:
    model_dir: str
    model_timestamp: str | None
    seq_len: int
    input_issues: list[str]
    last_issue: str
    predicted_red_balls: list[int]
    predicted_blue_ball: int
    normalized: list[float]
```

---

## `models/lstm.py`

### `LotteryLSTM(input_size=7, hidden_size=64, num_layers=2)`

```
LSTM(input_size, hidden_size, num_layers, batch_first=True)
Linear(hidden_size, 7)
Sigmoid
```

- `forward(x: (batch, seq_len, 7)) → (batch, 7)`
- 取 LSTM 最后时间步输出，经全连接层 + Sigmoid

---

## `training/trainer.py`

### `Trainer(model, config)`

- 自动选择 device（cuda / mps / cpu）
- 创建 SummaryWriter，按时间戳命名日志目录
- `train(loader, epochs, val_loader=None) → dict` 返回训练摘要

---

## `inference/loader.py`

### `load_model_artifact(model_path: str) → (LotteryLSTM, ModelArtifact)`

- 解析 `--model` 为产物目录（目录或 `model.pt` 文件）
- 加载 `metadata.json` 与 `model.pt`（state_dict）
- 返回处于 `eval` 模式的模型实例

---

## `inference/predictor.py`

### `predict_next(model, records, model_dir, metadata, seq_len=10) → PredictionResult`

- 取最近 `seq_len` 期构造输入张量 `(1, seq_len, 7)`
- 前向推理并反归一化为球号
- 返回 `PredictionResult`（含输入期号窗口与预测结果）

---

## `inference/saver.py`

### `save_prediction(result, summaries_dir, timestamp) → str`

- 将 `PredictionResult.to_dict()` 写入 `output/summaries/{timestamp}_prediction.json`

---

## `training/saver.py`

### `save_model(model, config, summary, timestamp) → str`

- 保存 `model.pt`（state_dict）
- 保存 `metadata.json`（时间戳、模型类型、summary、训练配置）
- 返回保存目录路径

---

## `cli/main.py`

### `main() → int`

- `data` 子命令：DuckDB 同步与状态（见 `cli/data.py`）
- `train` 子命令：完整训练流程
- `predict` 子命令：加载模型、执行推理、输出 JSON 并保存摘要
