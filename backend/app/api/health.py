# -*- coding: utf-8 -*-
"""backend/app/api/health.py — Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "OceanEmbed API"}
