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

## `training/saver.py`

### `save_model(model, config, summary, timestamp) → str`

- 保存 `model.pt`（state_dict）
- 保存 `metadata.json`（时间戳、模型类型、summary、训练配置）
- 返回保存目录路径

---

## `cli/main.py`

### `main() → int`

- `train` 子命令：完整训练流程
- `predict` 子命令：占位，打印消息退出
