"""PyTorch 数据集"""

import torch
from torch.utils.data import Dataset
from lottery.domain.types import LotteryRecord

RED_BALL_MAX = 33
BLUE_BALL_MAX = 16


def normalize_record(record: LotteryRecord) -> list[float]:
    """将单期开奖记录归一化为 7 维向量（红球×6 + 蓝球×1）。"""
    return [r / RED_BALL_MAX for r in record.red_balls] + [
        record.blue_ball / BLUE_BALL_MAX
    ]


def build_sequence_tensor(
    records: list[LotteryRecord], start_idx: int, seq_len: int
) -> torch.Tensor:
    """构造 seq_len 期滑动窗口特征张量，shape (1, seq_len, 7)。"""
    features = [
        normalize_record(records[start_idx + i]) for i in range(seq_len)
    ]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def denormalize_prediction(normalized: list[float]) -> tuple[list[int], int]:
    """将模型输出的归一化向量还原为球号。"""
    red = [
        max(1, min(RED_BALL_MAX, round(v * RED_BALL_MAX)))
        for v in normalized[:6]
    ]
    blue = max(1, min(BLUE_BALL_MAX, round(normalized[6] * BLUE_BALL_MAX)))
    return sorted(red), blue


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
        features = [
            normalize_record(self.records[i])
            for i in range(idx, idx + self.seq_len)
        ]

        target_record = self.records[idx + self.seq_len]
        target = normalize_record(target_record)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )
