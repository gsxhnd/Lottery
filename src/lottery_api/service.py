"""模型选择与预测服务"""

from datetime import datetime
from pathlib import Path

from lottery.config import load_config
from lottery.data import load_lottery_data
from lottery.inference import (
    DEFAULT_SEQ_LEN,
    load_model_artifact,
    predict_next,
    save_prediction,
)
from lottery.models import LotteryLSTM


class PredictionService:
    """管理配置、数据缓存与当前加载的模型。"""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self._config = load_config(config_path)
        self._records = load_lottery_data(self._config["data"]["raw_file"])
        self._model: LotteryLSTM | None = None
        self._model_dir: str | None = None
        self._metadata: dict | None = None

    @property
    def config(self) -> dict:
        return self._config

    @property
    def record_count(self) -> int:
        return len(self._records)

    def reload_data(self) -> int:
        """重新读取历史开奖数据，返回记录条数。"""
        self._records = load_lottery_data(self._config["data"]["raw_file"])
        return len(self._records)

    def list_models(self) -> list[dict]:
        """列出 models 目录下所有可用模型产物。"""
        models_dir = Path(self._config["output"]["models_dir"])
        if not models_dir.is_dir():
            return []

        items: list[dict] = []
        for entry in sorted(models_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            if not (entry / "model.pt").exists():
                continue
            rel_id = entry.name
            timestamp = None
            metadata_path = entry / "metadata.json"
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    meta = json.load(f)
                timestamp = meta.get("timestamp")
            items.append(
                {
                    "id": rel_id,
                    "path": str(entry.resolve()),
                    "timestamp": timestamp,
                }
            )
        return items

    def current_model(self) -> dict | None:
        """返回当前已加载模型信息。"""
        if self._model_dir is None:
            return None
        return {
            "path": self._model_dir,
            "timestamp": (self._metadata or {}).get("timestamp"),
        }

    def load_model(self, model_path: str) -> dict:
        """加载并缓存指定模型。"""
        model, artifact = load_model_artifact(model_path)
        self._model = model
        self._model_dir = artifact.model_path
        self._metadata = artifact.metadata
        return self.current_model() or {}

    def predict(
        self,
        model_path: str | None = None,
        *,
        save_summary: bool = False,
    ) -> dict:
        """执行下一期预测；可临时指定模型或使用已加载模型。"""
        if model_path is not None:
            self.load_model(model_path)

        if self._model is None or self._model_dir is None or self._metadata is None:
            raise RuntimeError("未加载模型，请先调用 POST /models/load 或在预测请求中指定 model")

        result = predict_next(
            self._model,
            self._records,
            model_dir=self._model_dir,
            metadata=self._metadata,
            seq_len=DEFAULT_SEQ_LEN,
        )
        payload = result.to_dict()
        summary_path = None
        if save_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = save_prediction(
                result,
                self._config["output"]["summaries_dir"],
                timestamp=timestamp,
            )
        payload["summary_path"] = summary_path
        return payload
