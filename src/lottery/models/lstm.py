"""PyTorch 模型定义"""

import torch
import torch.nn as nn


class LotteryLSTM(nn.Module):
    """基于 LSTM 的双色球预测模型"""

    def __init__(self, input_size: int = 7, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, 7)
        Returns:
            shape (batch, 7)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return torch.sigmoid(output)
