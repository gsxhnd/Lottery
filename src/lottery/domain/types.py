"""核心领域对象定义"""

from dataclasses import dataclass


@dataclass
class LotteryRecord:
    """单期开奖记录"""

    issue: str
    date: str
    red_balls: list[int]  # 排序后的红球
    blue_ball: int


@dataclass
class TrainingConfig:
    """训练配置"""

    epochs: int
    batch_size: int
    learning_rate: float


@dataclass
class ModelArtifact:
    """模型产物"""

    model_path: str
    metadata: dict
