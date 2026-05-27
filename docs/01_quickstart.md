# 快速开始

## 项目简介

`Lottery` 是一个基于 PyTorch LSTM 的双色球（SSQ）历史数据实验项目。项目定位为学习与工程练习，不提供投注建议。

当前已完成**最小训练闭环**：数据读取 → 样本构造 → 模型训练 → 产物输出。

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
# 正序（从早到晚）
curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt
# 或倒序（从晚到早）
curl -L "https://data.17500.cn/ssq_desc.txt" -o data/raw_ssq.txt
```

---

## 训练模型

```bash
# 使用默认配置
uv run lottery train

# 使用自定义配置
uv run lottery train --config my_config.toml
```

> 首次运行时如果缺少 `data/raw_ssq.txt`，会抛出 `FileNotFoundError`。

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
- [架构设计](./04_architecture.md)
