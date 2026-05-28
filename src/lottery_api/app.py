"""FastAPI 应用"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lottery_api.business.prediction_service import PredictionService
from lottery_api.handlers.routes import register_routes

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
    register_routes(app, get_service)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    if _STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def web_ui() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

    return app
