"""API 请求与响应模型"""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """已保存模型目录摘要"""

    id: str = Field(description="相对 models 目录的路径或目录名")
    path: str = Field(description="模型产物目录绝对路径")
    timestamp: str | None = Field(default=None, description="训练时间戳（metadata）")


class LoadModelRequest(BaseModel):
    """加载模型请求"""

    model: str = Field(description="模型目录或 model.pt 路径")


class PredictRequest(BaseModel):
    """预测请求（可选覆盖当前已加载模型）"""

    model: str | None = Field(
        default=None,
        description="模型路径；省略则使用当前已加载模型",
    )
    save_summary: bool = Field(
        default=False,
        description="是否将结果写入 output/summaries/",
    )


class PredictionCandidate(BaseModel):
    """单组备选号码"""

    red_balls: list[int]
    blue_ball: int
    hit_rate: float = Field(description="综合命中率（%），基于历史回测")
    red_hit_avg: float = Field(description="红球平均命中个数")
    blue_hit_rate: float = Field(description="蓝球命中率（%）")


class PredictionResponse(BaseModel):
    """预测结果"""

    model_dir: str
    model_timestamp: str | None
    input_window: dict
    candidates: list[PredictionCandidate]
    prediction: PredictionCandidate
    normalized: list[float]
    backtest_periods: int = Field(description="命中率回测使用的历史期数")
    summary_path: str | None = None


class DrawRecord(BaseModel):
    """单期开奖记录（供可视化接口使用）"""

    issue: str
    date: str
    red_balls: list[int]
    blue_ball: int
    red_sum: int


class BallFrequencyItem(BaseModel):
    """号码频次"""

    ball: int
    count: int


class WinningStatsResponse(BaseModel):
    """DuckDB 开奖统计数据"""

    total_records: int
    issue_range: dict[str, str | None]
    red_frequencies: list[BallFrequencyItem]
    blue_frequencies: list[BallFrequencyItem]
    recent_draws: list[DrawRecord]
