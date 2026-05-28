"""FastAPI 应用"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lottery_api.schemas import (
    BallFrequencyItem,
    DrawRecord,
    LoadModelRequest,
    ModelInfo,
    PredictRequest,
    PredictionResponse,
    WinningStatsResponse,
)
from lottery_api.service import PredictionService

_service: PredictionService | None = None
_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


def get_service() -> PredictionService:
    if _service is None:
        raise RuntimeError("PredictionService 未初始化")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    config_path = app.state.config_path
    _service = PredictionService(config_path=config_path)
    yield
    _service = None


def create_app(*, config_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Lottery Prediction API",
        description="双色球 LSTM 模型预测接口",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config_path = config_path

    @app.get("/health")
    def health() -> dict:
        svc = get_service()
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
        count = get_service().reload_data()
        return {"records": count}

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

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    if _STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def web_ui() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

    return app
