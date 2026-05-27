"""训练流程"""

from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """训练器"""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def train(self, train_loader: DataLoader, epochs: int) -> dict:
        """训练模型

        Returns:
            训练摘要
        """
        self.model.train()

        for epoch in range(epochs):
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
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        self.writer.close()
        return {"final_loss": avg_loss, "epochs": epochs}
