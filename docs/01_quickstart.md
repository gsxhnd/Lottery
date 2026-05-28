# 快速开始

## 项目简介

`Lottery` 是一个基于 PyTorch LSTM 的双色球（SSQ）历史数据实验项目。项目定位为学习与工程练习，不提供投注建议。

当前已完成**最小训练闭环**：原始数据 → DuckDB 清洗入库 → 样本构造 → 模型训练 → 产物输出。

---

## 环境要求

- Python >= 3.12
- `uv` 包管理器

```bash
uv sync
```

---

## 准备数据

```bash
mkdir -p data
# 正序（从早到晚，推荐）
curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt
# 或倒序（从晚到早）
curl -L "https://data.17500.cn/ssq_desc.txt" -o data/raw_ssq.txt
```

### 导入 DuckDB（推荐）

```bash
# 首次：全量导入
uv run lottery data sync --full

# 查看库状态
uv run lottery data status
```

后续更新 raw 文件后，执行 `uv run lottery data sync` 即可增量插入新期号。

---

## 训练模型

```bash
# 使用默认配置（source=auto 时从 DuckDB 读数据）
uv run lottery train

# 使用自定义配置
uv run lottery train --config my_config.toml
```

> 若未执行 `data sync` 且 DuckDB 不存在或为空，`source=auto` 会回退读取 `data/raw_ssq.txt`。若 raw 也不存在，会抛出 `FileNotFoundError`。

---

## 配置文件

复制示例配置并根据需要修改：

```bash
cp config/config.toml.example config/config.toml
```

详细配置项见 [配置指南](./03_config.md)。

---

## 下一步

- [CLI 命令参考](./02_cli.md)
- [配置指南](./03_config.md)
- [数据管道](./05_data_pipeline.md)
- [架构设计](./04_architecture.md)
