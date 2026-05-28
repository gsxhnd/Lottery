# 数据管道

## 概览

数据层负责将原始 SSQ 文本清洗入库（DuckDB），并在训练/推理时提供 `LotteryRecord` 列表与 PyTorch 样本。

```
data/raw_ssq.txt
    │  lottery data sync [--full]
    ▼
data/lottery.duckdb  (draws 表)
    │  load_lottery_records() / LotteryDataset.from_config()
    ▼
LotteryDataset → DataLoader → 训练
```

| 阶段 | 模块 | 说明 |
|------|------|------|
| 解析 | `data/parser.py` | 单行 raw → `LotteryRecord` |
| 入库 | `data/duckdb/store.py` | 全量/增量写入 DuckDB |
| 读取 | `data/repository.py` | 按 `source` 配置选择库或 raw |
| 样本 | `data/dataset.py` | 滑动窗口 + 归一化 |

---

## 原始数据格式

每行一期开奖记录，空格分隔：

```
2004001 2004-01-01 01 02 03 07 10 25 07 ...
```

| 列 | 字段 | 说明 | 示例 |
|----|------|------|------|
| 1 | 期号 | 开奖期数 | `2004001` |
| 2 | 日期 | ISO 日期 | `2004-01-01` |
| 3-8 | 红球（排序） | 按大小排序 | `01 02 03 07 10 25` |
| 9 | 蓝球 | 蓝球号码 | `07` |
| 10-15 | 红球（出球顺序） | 摇奖顺序 | — |
| 16+ | 扩展数据 | 销售额、奖池等 | — |

数据下载方式见 [快速开始](./01_quickstart.md#准备数据)。

**当前仅使用前 9 列**。列 10-15（出球顺序）和列 16+（扩展统计）暂不解析。

---

## 解析流程

实现位于 `data/parser.py`：

```
raw_ssq.txt
    ↓ iter_raw_records()
parse_raw_line(line)
    ↓ len(parts) >= 9
LotteryRecord(
    issue     = parts[0],
    date      = parts[1],
    red_balls = [int(parts[2..7])],
    blue_ball = int(parts[8])
)
```

`load_lottery_data(file_path)` 为兼容接口，内部调用 `iter_raw_records()`。

---

## DuckDB 存储

### 表结构

库文件默认路径：`data/lottery.duckdb`（见配置项 `data.db_file`）。

`draws` 表字段：

| 列 | 类型 | 说明 |
|----|------|------|
| `issue` | VARCHAR | 期号（主键） |
| `draw_date` | DATE | 开奖日期 |
| `red1` … `red6` | SMALLINT | 红球 |
| `blue` | SMALLINT | 蓝球 |
| `synced_at` | TIMESTAMP | 入库时间 |

### 同步命令

```bash
# 首次或需要重建时：清空表后全量导入
uv run lottery data sync --full

# 日常更新 raw 后：仅插入库中不存在的期号
uv run lottery data sync

# 查看库内条数与期号范围
uv run lottery data status
```

| 模式 | 行为 |
|------|------|
| 增量（默认） | 跳过已存在 `issue`，插入新期 |
| `--full` | `DELETE` 全表后从 raw 重导 |

详见 [CLI — data](./02_cli.md#data--duckdb-数据管道)。

### 编程接口

```python
from lottery.config import load_config
from lottery.data import sync_data, load_lottery_records, LotteryDataStore

config = load_config()

# raw → DuckDB
result = sync_data(config, full=False)  # SyncResult(inserted, skipped, total_in_db, mode)

# 读取全部记录（按期号升序）
records = load_lottery_records(config)

# 直接操作存储
store = LotteryDataStore(config["data"]["db_file"])
records = store.fetch_records()
```

---

## 训练/推理数据读取

`load_lottery_records(config)` 根据 `data.source` 决定来源：

| `source` | 行为 |
|----------|------|
| `auto`（默认） | DuckDB 存在且有条目 → 读库；否则读 `raw_file` |
| `duckdb` | 仅读库；库为空则报错并提示先 `data sync` |
| `raw` | 始终读原始文本，不查库 |

训练入口使用 `LotteryDataset.from_config(config, seq_len=10)`，内部调用 `load_lottery_records()`。

推理 CLI 与 `lottery-api` 的 `PredictionService` 同样通过 `load_lottery_records()` 加载历史数据。

---

## 样本构造

### LotteryDataset (PyTorch Dataset)

使用**滑动窗口**方式构造时序样本：

```python
seq_len = 10  # 用 10 期历史预测下一期

# 对每个 idx，构造：
features = records[idx .. idx+seq_len]     # 10 期特征
target   = records[idx+seq_len]            # 第 11 期目标
```

从配置构造：

```python
dataset = LotteryDataset.from_config(config, seq_len=10)
```

### 特征标准化

- 红球除以 33.0 → 归一化到 `[0, 1]`
- 蓝球除以 16.0 → 归一化到 `[0, 1]`
- 特征维度：`(seq_len, 7)` — 每期 6 个红球 + 1 个蓝球

### 数据集规模

- `__len__()` = `len(records) - seq_len`
- 无 train/val/test 划分，全量数据直接训练（验证集由 `training.val_split` 在 Dataset 上随机划分）

---

## 推荐工作流

```bash
mkdir -p data
curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt
uv run lottery data sync --full    # 首次入库
uv run lottery train               # 默认从 DuckDB 读数据

# 后续更新开奖数据
curl -L "https://data.17500.cn/ssq_asc.txt" -o data/raw_ssq.txt
uv run lottery data sync           # 增量
uv run lottery train
```

---

## 扩展计划

里程碑四将增加：

- 出球顺序特征（列 10-15）
- 配置开关控制特征选择
- 对比效果差异
