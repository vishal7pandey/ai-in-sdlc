"""Centralized exception handling for FastAPI."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("reqeng.errors")


async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail or "http_error", "message": exc.detail},
    )


async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "message": "Unexpected server error"},
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
