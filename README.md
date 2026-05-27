# Lottery

基于 PyTorch LSTM 的双色球（SSQ）历史数据实验项目。

> 项目定位是学习、实验与工程练习，不提供任何投注建议，也不承诺预测有效性。

## 当前状态

- ✅ 里程碑一：工程骨架
- ✅ 里程碑二：最小训练闭环（LSTM 训练 + TensorBoard）
- ⬜ 里程碑三：最小推理闭环

## 快速开始

```bash
uv sync                                # 安装依赖
mkdir -p data
curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt
uv run lottery train                   # 训练模型
```

详见 [快速开始](docs/usage/01_quickstart.md)。

## 文档

| 分类 | 内容 |
|------|------|
| [使用指南](docs/usage/) | 快速开始、CLI 参考、配置指南 |
| [开发文档](docs/dev/) | 路线图、设计原则、工程约定 |
| [代码原理](docs/wiki/) | 架构设计、数据管道、训练流程、模块接口 |

## 仓库结构

```text
Lottery/
├── data/                    # 数据目录（gitignore）
├── docs/                    # 文档
├── src/lottery/             # 源码
│   ├── cli/                 # 命令行入口
│   ├── config/              # 配置加载
│   ├── domain/              # 领域对象
│   ├── data/                # 数据处理
│   ├── models/              # 模型定义
│   ├── training/            # 训练流程
│   └── inference/           # 推理执行（占位）
├── config/
│   └── config.toml.example  # 配置示例
├── pyproject.toml
└── README.md
```
