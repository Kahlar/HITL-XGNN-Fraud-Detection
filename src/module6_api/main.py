"""FastAPI application initialization, middleware, exception handling, and router registration."""

from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.logger import get_logger
from src.module6_api.config import api_settings
from src.module6_api.dependencies import get_inference_service
from src.module6_api.routers import (
    analytics_router,
    explain_router,
    graph_router,
    health_router,
    hitl_router,
    transactions_router,
)
from src.module6_api.schemas.common import APIErrorResponse
from src.module8_database.connection import db_manager

logger = get_logger("Module6.Main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle event handler."""
    logger.info("=" * 80)
    logger.info(f"Starting {api_settings.app_title} v{api_settings.app_version}")
    logger.info(f"API Prefix: {api_settings.api_prefix} | Environment: {'Debug' if api_settings.debug else 'Production'}")
    logger.info("=" * 80)

    # Initialize inference service on startup
    inference_service = get_inference_service()
    if inference_service.is_available:
        logger.info(f"Active GNN Model: {inference_service.model_version} ready on device: {inference_service.device}")
    else:
        logger.warning("No active GNN model loaded at startup.")

    yield

    # Shutdown
    logger.info("Shutting down API service and closing connection pools...")
    await db_manager.close()


def create_app() -> FastAPI:
    """Factory function for configuring and creating the FastAPI application."""
    app = FastAPI(
        title=api_settings.app_title,
        description=api_settings.app_description,
        version=api_settings.app_version,
        lifespan=lifespan,
        docs_url=f"{api_settings.api_prefix}/docs",
        redoc_url=f"{api_settings.api_prefix}/redoc",
        openapi_url=f"{api_settings.api_prefix}/openapi.json",
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Timing & Logging Middleware
    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        if request.url.path.startswith(api_settings.api_prefix):
            logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({process_time_ms:.1f}ms)")
        return response

    # 3. Centralized Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=APIErrorResponse(
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=APIErrorResponse(
                error_code="VALIDATION_ERROR",
                message=str(exc),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled server exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIErrorResponse(
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred.",
                details=str(exc) if api_settings.debug else None,
            ).model_dump(mode="json"),
        )

    # 4. Include Routers
    api_prefix = api_settings.api_prefix
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(transactions_router, prefix=api_prefix)
    app.include_router(graph_router, prefix=api_prefix)
    app.include_router(explain_router, prefix=api_prefix)
    app.include_router(hitl_router, prefix=api_prefix)
    app.include_router(analytics_router, prefix=api_prefix)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.module6_api.main:app",
        host=api_settings.host,
        port=api_settings.port,
        reload=api_settings.debug,
    )
