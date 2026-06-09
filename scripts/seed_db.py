"""
Seed the database with a sample tenant and admin user.

Usage:
    python scripts/seed_db.py

Creates:
  - Tenant: "demo"
  - User: admin@demo.com / password123 (role: admin)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from app.core.database import async_session
from app.core.security import hash_password
from app.models.database import Tenant, User


async def seed():
    async with async_session() as db:
        # Check if tenant already exists
        result = await db.execute(select(Tenant).where(Tenant.name == "demo"))
        tenant = result.scalar_one_or_none()

        if tenant is None:
            tenant = Tenant(name="demo")
            db.add(tenant)
            await db.flush()
            print("✅ Created tenant: demo")
        else:
            print("ℹ️  Tenant 'demo' already exists")

        # Check if user already exists
        result = await db.execute(
            select(User).where(User.email == "admin@demo.com")
        )
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                tenant_id=tenant.tenant_id,
                email="admin@demo.com",
                hashed_password=hash_password("password123"),
                role="admin",
            )
            db.add(user)
            print("✅ Created user: admin@demo.com (role: admin, password: password123)")
        else:
            print("ℹ️  User 'admin@demo.com' already exists")

        await db.commit()

    print()
    print("Seed complete! You can now login with:")
    print("  POST /auth/login")
    print('  {"email": "admin@demo.com", "password": "password123"}')


if __name__ == "__main__":
    asyncio.run(seed())
