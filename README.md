# Lottery

本项目仅作为娱乐项目和学习作用，并没有实际参考意义。

## 历史中奖数据源

数据源：`http://e.17500.cn/getData/ssq.TXT`

- 期号
- 日期
- 红球1
- 红球2
- 红球3
- 红球4
- 红球5
- 红球6
- 蓝球
- 红球1（按摇号出现顺序）
- 红球2（按摇号出现顺序）
- 红球3（按摇号出现顺序）
- 红球4（按摇号出现顺序）
- 红球5（按摇号出现顺序）
- 红球6（按摇号出现顺序）

## How to run

```shell
# 下载数据
task data_download
## 创建 python 虚拟环境
python -m venv .venv
source .venv/bin/activate 
## 下载 python 需要的依赖
task mod
## 训练模型
task train
```
