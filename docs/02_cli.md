# CLI 命令参考

## 命令概览

```bash
uv run lottery [command] [options]
```

| 命令 | 说明 | 状态 |
|------|------|------|
| `data` | DuckDB 数据同步与状态 | ✅ 可用 |
| `train` | 训练模型 | ✅ 可用 |
| `predict` | 执行推理 | ✅ 可用 |

---

## `data` — DuckDB 数据管道

```bash
uv run lottery data {sync|status} [options]
```

### `data sync` — 同步 raw 到 DuckDB

```bash
uv run lottery data sync [--config PATH] [--full]
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--config` | 否 | TOML 配置文件路径 |
| `--full` | 否 | 全量重建（清空表后重导）；默认增量 |

**增量模式**：遍历 `data.raw_file`，跳过库中已有 `issue`，仅插入新期。

**全量模式**：`DELETE` 后从 raw 完整重导，适用于修正历史数据或首次建库。

示例：

```bash
uv run lottery data sync --full   # 首次建库
uv run lottery data sync          # 更新 raw 后增量
```

### `data status` — 查看 DuckDB 状态

```bash
uv run lottery data status [--config PATH]
```

输出：库路径、记录条数、首期/末期期号与日期。

---

## `train` — 训练模型

```bash
uv run lottery train [--config PATH]
```

### 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--config` | 否 | TOML 配置文件路径，默认 `config/config.toml` |

### 训练流程

1. 加载配置（文件或默认值）
2. 通过 `LotteryDataset.from_config()` 加载数据（`source=auto` 时优先 DuckDB）
3. 执行训练循环
4. 保存模型产物和 TensorBoard 日志

> 使用 DuckDB 前需先执行 `uv run lottery data sync`（或 `--full`）。若库不存在或为空且 `source=auto`，会回退读取 `raw_file`。

### 示例

```bash
uv run lottery data sync --full
uv run lottery train
uv run lottery train --config prod.toml
```

---

## `predict` — 执行推理

```bash
uv run lottery predict --model PATH [--config PATH]
```

### 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--model` | 是 | 模型产物目录，或 `model.pt` 文件路径 |
| `--config` | 否 | TOML 配置文件路径，默认 `config/config.toml` |

### 推理流程

1. 加载配置（用于数据路径与输出目录）
2. 加载 `model.pt` 与 `metadata.json`
3. 通过 `load_lottery_records()` 读取历史数据，取最近 `seq_len=10` 期作为输入
4. 执行前向推理并反归一化球号
5. 打印 JSON 结果，并写入 `output/summaries/{timestamp}_prediction.json`

### 示例

```bash
uv run lottery predict --model output/models/20250528_120000
uv run lottery predict --model output/models/20250528_120000/model.pt
```

---

## 帮助

```bash
uv run lottery --help
uv run lottery data --help
uv run lottery data sync --help
uv run lottery train --help
uv run lottery predict --help
```
