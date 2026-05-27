"""PyTorch 数据集"""

import torch
from torch.utils.data import Dataset
from lottery.domain.types import LotteryRecord


class LotteryDataset(Dataset):
    """双色球数据集"""

    def __init__(self, records: list[LotteryRecord], seq_len: int = 10):
        """
        Args:
            records: 开奖记录列表
            seq_len: 序列长度
        """
        self.records = records
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.records) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取一个样本

        Returns:
            (features, target): 特征和目标
        """
        # 特征：seq_len 期的红球+蓝球，shape (seq_len, 7)，归一化
        features = []
        for i in range(idx, idx + self.seq_len):
            record = self.records[i]
            normalized = [r / 33.0 for r in record.red_balls] + [
                record.blue_ball / 16.0
            ]
            features.append(normalized)

        # 目标：下一期的红球+蓝球（归一化）
        target_record = self.records[idx + self.seq_len]
        target = [r / 33.0 for r in target_record.red_balls] + [
            target_record.blue_ball / 16.0
        ]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )
