"""模型产物保存"""

import json
from pathlib import Path
from datetime import datetime
import torch


def save_model(
    model: torch.nn.Module, config: dict, summary: dict, timestamp: str
) -> str:
    """保存模型产物

    Returns:
        保存路径
    """
    model_dir = Path(config["output"]["models_dir"]) / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    model_path = model_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # 保存元数据
    metadata = {
        "timestamp": timestamp,
        "model_type": model.__class__.__name__,
        "summary": summary,
        "config": config["training"],
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(model_dir)
