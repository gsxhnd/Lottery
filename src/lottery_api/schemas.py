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


class PredictionResponse(BaseModel):
    """预测结果"""

    model_dir: str
    model_timestamp: str | None
    input_window: dict
    prediction: dict
    normalized: list[float]
    summary_path: str | None = None
