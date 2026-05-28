"""核心领域对象定义"""

from dataclasses import dataclass
from lottery_data import LotteryRecord


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


@dataclass
class PredictionResult:
    """推理结果"""

    model_dir: str
    model_timestamp: str | None
    seq_len: int
    input_issues: list[str]
    last_issue: str
    predicted_red_balls: list[int]
    predicted_blue_ball: int
    normalized: list[float]

    def to_dict(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "model_timestamp": self.model_timestamp,
            "input_window": {
                "seq_len": self.seq_len,
                "issues": self.input_issues,
                "last_issue": self.last_issue,
            },
            "prediction": {
                "red_balls": self.predicted_red_balls,
                "blue_ball": self.predicted_blue_ball,
            },
            "normalized": self.normalized,
        }
