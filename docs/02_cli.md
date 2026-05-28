# CLI 命令参考

## 命令概览

```bash
uv run lottery [command] [options]
```

| 命令 | 说明 | 状态 |
|------|------|------|
| `train` | 训练模型 | ✅ 可用 |
| `predict` | 执行推理 | ✅ 可用 |

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
2. 读取数据并构造样本
3. 执行训练循环
4. 保存模型产物和 TensorBoard 日志

### 示例

```bash
# 默认配置训练
uv run lottery train

# 自定义配置
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
3. 读取历史数据，取最近 `seq_len=10` 期作为输入
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
uv run lottery train --help
uv run lottery predict --help
```
