"""训练流程"""

import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """训练器"""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = self._resolve_device(config)
        self.model.to(self.device)

        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )
        self.criterion = nn.MSELoss()

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(config["output"]["logs_dir"]) / timestamp
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.timestamp = timestamp

    def _resolve_device(self, config: dict) -> torch.device:
        """解析训练设备，支持显式配置和自动选择。"""
        configured = config.get("training", {}).get("device")
        if configured:
            return torch.device(configured)

        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def train(
        self, train_loader: DataLoader, epochs: int, val_loader: DataLoader | None = None
    ) -> dict:
        """训练模型

        Returns:
            训练摘要
        """
        final_val_loss = None
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (features, targets) in enumerate(train_loader):
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # 记录到 TensorBoard
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar("Loss/train", loss.item(), global_step)

            avg_loss = total_loss / len(train_loader)
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)

            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                final_val_loss = val_loss
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        self.writer.close()
        summary = {"final_loss": avg_loss, "epochs": epochs}
        if final_val_loss is not None:
            summary["final_val_loss"] = final_val_loss
        return summary

    def _evaluate(self, val_loader: DataLoader) -> float:
        """在验证集上计算平均损失。"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)
