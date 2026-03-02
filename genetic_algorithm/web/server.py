"""
FastAPI application factory for the GA Web Dashboard.

Creates the app with:
  - CORS middleware (for React dev server on :5173)
  - Lifespan context manager (RunManager lifecycle)
  - All REST API routers
  - WebSocket endpoint
  - Static file serving for the React build (production)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from genetic_algorithm.web.config import WebConfig
from genetic_algorithm.web.run_manager import RunManager
from genetic_algorithm.web.services.data_service import DataService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("GA Dashboard starting up")
    yield
    logger.info("GA Dashboard shutting down")


def create_app(
    web_config: Optional[WebConfig] = None,
    run_manager: Optional[RunManager] = None,
) -> FastAPI:
    """
    Build and return the FastAPI application.

    Args:
        web_config: Dashboard configuration (defaults applied if None).
        run_manager: Shared RunManager instance (created if None).

    Returns:
        Configured FastAPI app ready to run with uvicorn.
    """
    web_config = web_config or WebConfig()

    app = FastAPI(
        title="GA Evolution Dashboard",
        description="Real-time monitoring and control for the Genetic Algorithm trading strategy engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=web_config.cors_origins + ["*"],  # Permissive for local dev
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Shared state ───────────────────────────────────────────────
    mgr = run_manager or RunManager()
    data_svc = DataService(run_manager=mgr)

    app.state.web_config = web_config
    app.state.run_manager = mgr
    app.state.data_service = data_svc

    # ── API Routers ────────────────────────────────────────────────
    from genetic_algorithm.web.routers import (
        ws,
        runs,
        generations,
        strategies,
        config,
        backtest,
        data,
        dry_run,
    )

    app.include_router(ws.router)
    app.include_router(runs.router)
    app.include_router(generations.router)
    app.include_router(strategies.router)
    app.include_router(config.router)
    app.include_router(backtest.router)
    app.include_router(data.router)
    app.include_router(dry_run.router)

    # ── Health check ───────────────────────────────────────────────
    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "active_runs": len([
                r for r in mgr.list_runs()
                if r.status.value in ("running", "paused")
            ]),
        }

    # ── Static files (React build) ────────────────────────────────
    frontend_build = Path(__file__).parent / "frontend" / "dist"
    if frontend_build.exists():
        app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")
        logger.info("Serving frontend from %s", frontend_build)
    else:
        logger.info(
            "No frontend build found at %s — run 'npm run build' in web/frontend/",
            frontend_build,
        )
        # Serve a simple placeholder
        @app.get("/")
        async def root():
            return {
                "message": "GA Dashboard API is running. Frontend not built yet.",
                "docs": "/docs",
                "api": "/api/health",
            }

    return app


def start_server(
    web_config: Optional[WebConfig] = None,
    run_manager: Optional[RunManager] = None,
) -> None:
    """
    Start the dashboard server (blocking).

    Typically called from the CLI entry point.
    """
    import uvicorn

    web_config = web_config or WebConfig()
    app = create_app(web_config=web_config, run_manager=run_manager)

    logger.info("Starting dashboard at http://%s:%d", web_config.host, web_config.port)

    # Open browser automatically if configured
    if web_config.open_browser:
        import threading
        import webbrowser

        url = f"http://{web_config.host}:{web_config.port}"
        # Delay slightly so the server has time to start
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    uvicorn.run(
        app,
        host=web_config.host,
        port=web_config.port,
        log_level=web_config.log_level,
    )
