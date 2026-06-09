"""
Initialize the database — create all tables.

Usage:
    python scripts/init_db.py

This is a convenience script for local development.
In production, use Alembic migrations: alembic upgrade head
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import engine
from app.models.database import Base


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("✅ Database tables created successfully!")
    print()
    print("Tables created:")
    for table_name in Base.metadata.tables:
        print(f"  - {table_name}")


if __name__ == "__main__":
    asyncio.run(init_db())
