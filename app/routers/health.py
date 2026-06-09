"""
Health check endpoint — verifies all service dependencies.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.retriever import retriever_service

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    System health check — verifies API, PostgreSQL, and Qdrant.
    """
    services = {
        "api": "up",
        "postgres": "down",
        "qdrant": "down",
    }

    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        services["postgres"] = "up"
    except Exception:
        services["postgres"] = "down"

    # Check Qdrant
    try:
        if retriever_service.is_healthy():
            services["qdrant"] = "up"
    except Exception:
        services["qdrant"] = "down"

    # Determine overall status
    all_up = all(v == "up" for v in services.values())
    status = "healthy" if all_up else "degraded"

    return {
        "success": True,
        "data": {
            "status": status,
            "version": "2.0.0",
            "services": services,
        },
        "error": None,
    }
