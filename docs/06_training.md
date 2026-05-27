# 训练流程

## 训练入口

```
cli: _train()
  → Trainer(model, config)
  → trainer.train(train_loader, epochs)
  → save_model(...)
```

## Trainer 类

位于 `training/trainer.py`：

```python
class Trainer:
    def __init__(self, model, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = Adam(model.parameters(), lr)
        self.criterion = MSELoss()
        self.writer = SummaryWriter(log_dir)
        self.timestamp = datetime.now()
```

### 训练循环

```
for epoch in range(epochs):
    for batch in train_loader:
        features, targets → device
        outputs = model(features)
        loss = MSELoss(outputs, targets)
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss, global_step)
    writer.add_scalar("Loss/epoch", avg_loss, epoch)
```

### 关键参数

| 参数 | 默认值 | 来源 |
|------|--------|------|
| `epochs` | 100 | `config/config.toml` → `training.epochs` |
| `batch_size` | 32 | `config/config.toml` → `training.batch_size` |
| `learning_rate` | 0.001 | `config/config.toml` → `training.learning_rate` |
| `seq_len` | 10 | **硬编码** (`cli/main.py`) |
| `hidden_size` | 64 | LSTM 默认参数 |
| `num_layers` | 2 | LSTM 默认参数 |

---

## 模型保存

位于 `training/saver.py`：

```
save_model(model, config, summary, timestamp)
  → output/models/{timestamp}/
      ├── model.pt          # state_dict
      └── metadata.json     # 时间戳、模型类型、summary、训练配置
```

---

## 损失函数

使用 **MSE Loss**（均方误差）：

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

其中 $y$ 和 $\hat{y}$ 都是归一化到 [0,1] 的 7 维向量（6 红球 + 1 蓝球）。

## 优化器

Adam（`lr=0.001`），使用 PyTorch 默认 `betas=(0.9, 0.999)`。

---

## TensorBoard

训练日志写入 `output/logs/{timestamp}/`：

| 指标 | 说明 |
|------|------|
| `Loss/train` | 每个 batch 的损失 |
| `Loss/epoch` | 每个 epoch 的平均损失 |

```bash
tensorboard --logdir=output/logs
# 访问 http://localhost:6006

# 指定端口
tensorboard --logdir=output/logs --port=6007
```
