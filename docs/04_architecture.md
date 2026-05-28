# 架构与设计

## 设计目标

项目定位为学习型工程实验，优先追求结构清晰、可维护、可复现，而不是模型预测效果。

## 核心原则

1. **先做最小闭环** — 跑通最小训练链路，评估、推理和多模型扩展放在闭环稳定之后。
2. **先定边界，再写实现** — 先明确模块职责边界，再逐层落地代码。
3. **先可维护，再谈可扩展** — 第一版保持简单，新增能力应表现为"新增模块"而非"重写现有目录"。

## 第一版明确不做

- 不同时支持多种模型体系
- 不引入复杂配置框架
- 不把训练、推理、评估塞进同一个脚本
- 不为"未来可能用到"而过度设计抽象层

---

## 分层设计

```
┌─────────────────────────────────────────┐
│                  cli                     │  应用层：命令解析与流程编排
├─────────────────────────────────────────┤
│   training          │    inference       │  业务层：训练执行 / 推理执行
├─────────────────────────────────────────┤
│   models            │    data            │  能力层：模型定义 / 解析·入库·样本
├─────────────────────────────────────────┤
│   domain                                │  领域层：公共数据对象
├─────────────────────────────────────────┤
│   config                                │  配置层：配置加载与校验
└─────────────────────────────────────────┘
```

### 模块职责

| 模块 | 负责 | 不负责 |
|------|------|--------|
| `config` | 加载、合并、校验配置 | 业务流程编排 |
| `domain` | 定义公共数据对象 | IO 和命令调用 |
| `data` | 解析 raw、DuckDB 同步、构造 PyTorch 样本 | 模型训练控制 |
| `models` | 构建模型实例 | 命令参数解析 |
| `training` | 执行训练、保存产物 | 直接处理 CLI 输入 |
| `inference` | 加载模型、执行预测 | 训练阶段内部状态 |
| `cli` | 接收用户命令、组织流程 | 承担底层业务实现 |

---

## 数据流

```
data/raw_ssq.txt
    │
    ▼  lottery data sync [--full]
data/lottery.duckdb (draws)
    │
    ▼  load_lottery_records()     ← data/repository.py
    │  (list[LotteryRecord])
    ▼
LotteryDataset.from_config()      ← data/dataset.py
    │  (seq_len=10 滑动窗口)
    ▼
DataLoader                        ← PyTorch
    │  (batch_size=32)
    ▼
LotteryLSTM                       ← models/lstm.py
    │  (input=7, hidden=64, layers=2)
    ▼
Trainer.train()                   ← training/trainer.py
    │  (MSE + Adam, TensorBoard)
    ▼
save_model()                      ← training/saver.py
    │  (model.pt + metadata.json)
    ▼
output/models/{timestamp}/
    │
    ▼
load_model_artifact()             ← inference/loader.py
    │
    ▼
predict_next()                    ← inference/predictor.py
    │
    ▼
output/summaries/{timestamp}_prediction.json
```

`source=raw` 或 `auto` 且库为空时，`load_lottery_records()` 直接从 `raw_ssq.txt` 读取，跳过 DuckDB。

## 调用链路

```
cli/main.py: data sync
  ├── load_config()                     → config/loader.py
  └── sync_data(config, full=...)       → data/repository.py → data/duckdb/store.py

cli/main.py: _train()
  ├── load_config(config_path)          → config/loader.py
  ├── LotteryDataset.from_config()      → data/dataset.py → load_lottery_records()
  ├── DataLoader(dataset, batch_size)   → PyTorch
  ├── LotteryLSTM()                     → models/lstm.py
  ├── Trainer(model, config)            → training/trainer.py
  └── save_model(model, config, ...)    → training/saver.py

cli/main.py: _predict()
  ├── load_config(config_path)          → config/loader.py
  ├── load_model_artifact(model_path)   → inference/loader.py
  ├── load_lottery_records(config)      → data/repository.py
  ├── predict_next(model, records, ...) → inference/predictor.py
  └── save_prediction(result, ...)      → inference/saver.py
```

---

## LSTM 模型结构

```
输入: (batch, 10, 7)   — 10 期历史，每期 7 维（红球×6 + 蓝球×1）
  │
  ▼
LSTM(input=7, hidden=64, layers=2)
  │
  ▼
取最后时刻输出 → Linear(64, 7) → Sigmoid
  │
  ▼
输出: (batch, 7)        — 下一期预测（红球×6 + 蓝球×1，归一化到 [0,1]）
```
