# 模块接口

## 模块依赖图

```
cli/main.py
 ├── config/loader.py     → load_config()
 ├── data/loader.py       → load_lottery_data()
 ├── data/dataset.py      → LotteryDataset
 ├── models/lstm.py       → LotteryLSTM
 └── training/
      ├── trainer.py      → Trainer
      └── saver.py        → save_model()
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
    "data": {"raw_file": "data/raw_ssq.txt"},
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
    },
}
```

---

## `data/loader.py`

### `load_lottery_data(file_path: str) → list[LotteryRecord]`

- 读取空格分隔的原始数据
- 跳过不足 9 列的行
- 解析期号、日期、红球（列 2-7）、蓝球（列 8）

---

## `data/dataset.py`

### `LotteryDataset(records, seq_len=10)`

- `__len__()` → `len(records) - seq_len`
- `__getitem__(idx)` → `(features, target)`
  - `features`: `(seq_len, 7)` 归一化序列
  - `target`: `(7,)` 归一化下一期

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

- 自动选择 device（cuda / cpu）
- 创建 SummaryWriter，按时间戳命名日志目录
- `train(loader, epochs) → dict` 返回 `{"final_loss", "epochs"}`

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

- `train` 子命令：完整训练流程
- `predict` 子命令：加载模型、执行推理、输出 JSON 并保存摘要
