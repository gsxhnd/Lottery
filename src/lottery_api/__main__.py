"""启动预测 API 服务：uv run lottery-api"""

import argparse


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Lottery 预测 API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    from lottery_api.app import create_app

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
