"""React SPA 静态资源托管与前端路由回退。"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"

# 这些路径前缀不走 index.html 回退（由 API 或 FastAPI 内置路由处理）
_SPA_EXCLUDED_PREFIXES = (
    "api/",
    "api",
    "docs",
    "redoc",
    "openapi.json",
)


def resolve_static_dir() -> Path | None:
    if (_STATIC_DIR / "index.html").is_file():
        return _STATIC_DIR
    return None


def _is_excluded_spa_path(path: str) -> bool:
    normalized = path.strip("/")
    if not normalized:
        return False
    for prefix in _SPA_EXCLUDED_PREFIXES:
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            return True
    return False


def _safe_static_file(static_root: Path, relative_path: str) -> Path | None:
    candidate = (static_root / relative_path).resolve()
    try:
        if not candidate.is_relative_to(static_root):
            return None
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def mount_spa(app: FastAPI, static_dir: Path) -> None:
    """挂载构建产物：/assets 静态目录 + 其余路径回退到 index.html。"""
    static_root = static_dir.resolve()
    index_path = static_root / "index.html"

    assets_dir = static_root / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        if _is_excluded_spa_path(full_path):
            raise HTTPException(status_code=404, detail="Not Found")

        if full_path:
            file_path = _safe_static_file(static_root, full_path)
            if file_path is not None:
                return FileResponse(file_path)

        return FileResponse(index_path)
