# Lottery 文档

基于 PyTorch LSTM 的双色球（SSQ）历史数据实验项目。

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速开始](./01_quickstart.md) | 环境搭建、数据准备、DuckDB 入库、训练模型 |
| [CLI 命令参考](./02_cli.md) | `data` / `train` / `predict` 命令详解 |
| [配置指南](./03_config.md) | TOML 配置项说明与默认值 |
| [架构与设计](./04_architecture.md) | 设计原则、分层架构、数据流 |
| [数据管道](./05_data_pipeline.md) | raw 解析、DuckDB 同步、样本构造、训练读数 |
| [训练流程](./06_training.md) | Trainer、损失函数、TensorBoard |
| [模块接口](./07_modules.md) | 各模块 API 与依赖关系 |
| [工程约定](./08_conventions.md) | 包结构、输出目录、已知技术债 |
| [开发路线图](./09_roadmap.md) | 里程碑计划与当前进度 |
