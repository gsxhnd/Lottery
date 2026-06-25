"""模型选择与预测业务服务。"""

from datetime import datetime
from pathlib import Path

from lottery_train.config import load_config
from lottery_train.inference import (
    DEFAULT_SEQ_LEN,
    load_model_artifact,
    predict_next,
    save_prediction,
)
from lottery_train.models import LotteryLSTM
from lottery_api.data.draw_repository import DrawRepository


class PredictionService:
    """管理配置、数据缓存与当前加载的模型。"""

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self._config = load_config(config_path)
        self._draw_repo = DrawRepository(self._config)
        self._records = self._draw_repo.fetch_records()
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
        self._records = self._draw_repo.fetch_records()
        return len(self._records)

    def get_winning_stats(self, *, recent_limit: int = 120) -> dict:
        records = self._draw_repo.fetch_records()
        if not records:
            return {
                "total_records": 0,
                "issue_range": {"start": None, "end": None},
                "red_frequencies": [],
                "blue_frequencies": [],
                "recent_draws": [],
            }

        red_counts = {ball: 0 for ball in range(1, 34)}
        blue_counts = {ball: 0 for ball in range(1, 17)}
        for record in records:
            for ball in record.red_balls:
                red_counts[ball] += 1
            blue_counts[record.blue_ball] += 1

        recent_records = records[-max(recent_limit, 1) :]
        recent_draws = [
            {
                "issue": record.issue,
                "date": record.date,
                "red_balls": record.red_balls,
                "blue_ball": record.blue_ball,
                "red_sum": sum(record.red_balls),
            }
            for record in recent_records
        ]

        return {
            "total_records": len(records),
            "issue_range": {"start": records[0].issue, "end": records[-1].issue},
            "red_frequencies": [
                {"ball": ball, "count": count}
                for ball, count in sorted(red_counts.items())
            ],
            "blue_frequencies": [
                {"ball": ball, "count": count}
                for ball, count in sorted(blue_counts.items())
            ],
            "recent_draws": recent_draws,
        }

    def list_models(self) -> list[dict]:
        models_dir = Path(self._config["output"]["models_dir"])
        if not models_dir.is_dir():
            return []

        items: list[dict] = []
        for entry in sorted(models_dir.iterdir(), reverse=True):
            if not entry.is_dir() or not (entry / "model.pt").exists():
                continue
            timestamp = None
            metadata_path = entry / "metadata.json"
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    meta = json.load(f)
                timestamp = meta.get("timestamp")
            items.append(
                {
                    "id": entry.name,
                    "path": str(entry.resolve()),
                    "timestamp": timestamp,
                }
            )
        return items

    def current_model(self) -> dict | None:
        if self._model_dir is None:
            return None
        return {
            "path": self._model_dir,
            "timestamp": (self._metadata or {}).get("timestamp"),
        }

    def load_model(self, model_path: str) -> dict:
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
        payload["summary_path"] = None
        if save_summary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            payload["summary_path"] = save_prediction(
                result,
                self._config["output"]["summaries_dir"],
                timestamp=timestamp,
            )
        return payload
