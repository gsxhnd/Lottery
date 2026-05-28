"""推理结果保存"""

import json
from datetime import datetime
from pathlib import Path

from lottery.domain.types import PredictionResult


def save_prediction(
    result: PredictionResult, summaries_dir: str, timestamp: str | None = None
) -> str:
    """将推理结果写入 summaries 目录。

    Returns:
        保存的文件路径
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(summaries_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{timestamp}_prediction.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return str(out_path)
