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
class PredictionCandidate:
    """单组备选号码"""

    red_balls: list[int]
    blue_ball: int
    hit_rate: float = 0.0
    red_hit_avg: float = 0.0
    blue_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "red_balls": self.red_balls,
            "blue_ball": self.blue_ball,
            "hit_rate": self.hit_rate,
            "red_hit_avg": self.red_hit_avg,
            "blue_hit_rate": self.blue_hit_rate,
        }


@dataclass
class PredictionResult:
    """推理结果"""

    model_dir: str
    model_timestamp: str | None
    seq_len: int
    input_issues: list[str]
    last_issue: str
    candidates: list[PredictionCandidate]
    normalized: list[float]
    backtest_periods: int = 0

    @property
    def predicted_red_balls(self) -> list[int]:
        return self.candidates[0].red_balls

    @property
    def predicted_blue_ball(self) -> int:
        return self.candidates[0].blue_ball

    def to_dict(self) -> dict:
        candidate_dicts = [c.to_dict() for c in self.candidates]
        return {
            "model_dir": self.model_dir,
            "model_timestamp": self.model_timestamp,
            "input_window": {
                "seq_len": self.seq_len,
                "issues": self.input_issues,
                "last_issue": self.last_issue,
            },
            "candidates": candidate_dicts,
            "prediction": candidate_dicts[0],
            "normalized": self.normalized,
            "backtest_periods": self.backtest_periods,
        }
