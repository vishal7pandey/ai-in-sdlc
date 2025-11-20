"""FastAPI application entrypoint for the Requirements Engineering Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.error_handler import setup_exception_handlers
from src.api.middleware.logging import logging_middleware
from src.api.routes import auth, health, rd, sessions, websocket
from src.config import settings
from src.storage.postgres import init_database
from src.storage.redis_cache import init_redis
from src.storage.vectorstore import init_vector_store

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("reqeng.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Requirements Engineering Platform")

    await init_database()
    await init_redis()
    await init_vector_store()

    logger.info("✅ All services initialized")
    yield
    logger.info("🛑 Shutting down gracefully")


app = FastAPI(
    title="Requirements Engineering Platform API",
    version=settings.APP_VERSION,
    description="Multi-agent conversational requirements engineering platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)
setup_exception_handlers(app)

app.include_router(health.router, tags=["System"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
app.include_router(rd.router, prefix="/api/v1/rd", tags=["Requirements Documents"])
app.include_router(websocket.router, tags=["WebSocket"])


@app.get("/")
async def root():
    return {
        "message": "Requirements Engineering Platform API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
