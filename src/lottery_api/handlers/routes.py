"""HTTP handler 层。"""

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from lottery_api.business.prediction_service import PredictionService
from lottery_api.schemas import (
    BallFrequencyItem,
    DrawRecord,
    LoadModelRequest,
    ModelInfo,
    PredictRequest,
    PredictionResponse,
    WinningStatsResponse,
)


def register_routes(app: FastAPI, get_service) -> None:
    @app.get("/health")
    def health() -> dict:
        svc: PredictionService = get_service()
        return {
            "status": "ok",
            "records": svc.record_count,
            "model_loaded": svc.current_model() is not None,
        }

    @app.get("/models", response_model=list[ModelInfo])
    def list_models() -> list[ModelInfo]:
        return [ModelInfo(**item) for item in get_service().list_models()]

    @app.get("/models/current")
    def current_model() -> dict:
        info = get_service().current_model()
        if info is None:
            raise HTTPException(status_code=404, detail="当前未加载模型")
        return info

    @app.post("/models/load")
    def load_model(body: LoadModelRequest) -> dict:
        try:
            return get_service().load_model(body.model)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictRequest) -> PredictionResponse:
        try:
            payload = get_service().predict(
                body.model,
                save_summary=body.save_summary,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return PredictionResponse(
            model_dir=payload["model_dir"],
            model_timestamp=payload["model_timestamp"],
            input_window=payload["input_window"],
            prediction=payload["prediction"],
            normalized=payload["normalized"],
            summary_path=payload.get("summary_path"),
        )

    @app.post("/predict/quick", response_model=PredictionResponse)
    def predict_quick(
        model: Annotated[str, Query(description="模型目录或 model.pt 路径")],
        save_summary: bool = False,
    ) -> PredictionResponse:
        return predict(PredictRequest(model=model, save_summary=save_summary))

    @app.post("/data/reload")
    def reload_data() -> dict:
        return {"records": get_service().reload_data()}

    @app.get("/data/winning-stats", response_model=WinningStatsResponse)
    def winning_stats(
        recent_limit: Annotated[
            int, Query(ge=10, le=500, description="最近开奖条数")
        ] = 120,
    ) -> WinningStatsResponse:
        payload = get_service().get_winning_stats(recent_limit=recent_limit)
        return WinningStatsResponse(
            total_records=payload["total_records"],
            issue_range=payload["issue_range"],
            red_frequencies=[BallFrequencyItem(**item) for item in payload["red_frequencies"]],
            blue_frequencies=[
                BallFrequencyItem(**item) for item in payload["blue_frequencies"]
            ],
            recent_draws=[DrawRecord(**item) for item in payload["recent_draws"]],
        )
