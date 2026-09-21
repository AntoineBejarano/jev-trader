from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from src.api.app import create_app
from src.config.settings import load_settings
from src.startup.preflight import PreflightError, run_preflight
from src.trader.engine import TraderEngine


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


async def run() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("src.main")
    try:
        await run_preflight(settings)
    except PreflightError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    engine = TraderEngine(settings)
    app = create_app(engine)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level="warning",
            access_log=False,
        )
    )
    logger.info(
        "Operator dashboard http://%s:%s/dashboard",
        settings.dashboard_host,
        settings.dashboard_port,
    )
    await engine.start_strategy()
    try:
        await server.serve()
    finally:
        await engine.shutdown()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
