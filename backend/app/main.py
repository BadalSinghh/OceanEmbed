# -*- coding: utf-8 -*-
"""
backend/app/main.py
===================
FastAPI application for OceanEmbed.
Uses backend/app/loaders.py for all data/model access (no Streamlit required).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import health, models, predict, results

app = FastAPI(
    title="OceanEmbed API",
    description=(
        "REST API for OceanEmbed — a deep learning framework for subsurface "
        "ocean temperature reconstruction from satellite surface observations "
        "over the Bay of Bengal."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS env var (comma-separated) to add production origins.
# Example: ALLOWED_ORIGINS=https://ocean-embed.vercel.app
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
_extra = [o.strip() for o in _env_origins.split(",") if o.strip()]

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ocean-embed-alpha.vercel.app",
    "https://ocean-embed.vercel.app",
] + _extra

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(results.router, prefix="/api")


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "service": "OceanEmbed API",
        "docs": "/api/docs",
        "health": "/api/health",
    }
