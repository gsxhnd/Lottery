"""模型产物加载"""

import json
from pathlib import Path

import torch

from lottery_train.domain.types import ModelArtifact
from lottery_train.models import LotteryLSTM


def resolve_model_dir(model_path: str) -> Path:
    """将 --model 参数解析为产物目录（含 model.pt 与 metadata.json）。"""
    path = Path(model_path)
    if path.is_file():
        if path.name != "model.pt":
            raise FileNotFoundError(f"期望 model.pt，实际为: {path}")
        return path.parent
    if path.is_dir():
        if not (path / "model.pt").exists():
            raise FileNotFoundError(f"目录中未找到 model.pt: {path}")
        return path
    raise FileNotFoundError(f"模型路径不存在: {model_path}")


def load_model_artifact(model_path: str) -> tuple[LotteryLSTM, ModelArtifact]:
    """加载已保存的模型与元数据。"""
    model_dir = resolve_model_dir(model_path)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到 metadata.json: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    model = LotteryLSTM()
    state_path = model_dir / "model.pt"
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    artifact = ModelArtifact(model_path=str(model_dir), metadata=metadata)
    return model, artifact
