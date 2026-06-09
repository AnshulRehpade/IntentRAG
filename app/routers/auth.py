"""
Authentication endpoints — registration and JWT token management.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import CurrentUser, get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.models.database import Tenant, User

router = APIRouter()


# --- Request / Response Schemas ---


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_name: str
    role: str = "reader"  # admin | reader | writer


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# --- Endpoints ---


@router.post("/register")
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user. Creates the tenant if it doesn't exist.
    First user in a tenant gets 'admin' role regardless of request.
    """
    try:
        # Check if email already taken
        existing = await db.execute(
            select(User).where(User.email == request.email)
        )
        if existing.scalar_one_or_none():
            return {
                "success": False,
                "data": None,
                "error": "Email already registered",
            }

        # Find or create tenant
        tenant_result = await db.execute(
            select(Tenant).where(Tenant.name == request.tenant_name)
        )
        tenant = tenant_result.scalar_one_or_none()

        if tenant is None:
            # New tenant — first user becomes admin
            tenant = Tenant(name=request.tenant_name)
            db.add(tenant)
            await db.flush()  # Get tenant_id
            role = "admin"
        else:
            role = request.role

        # Create user
        user = User(
            tenant_id=tenant.tenant_id,
            email=request.email,
            hashed_password=hash_password(request.password),
            role=role,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        # Generate token
        token = create_access_token(
            data={
                "sub": user.email,
                "user_id": str(user.user_id),
                "tenant_id": str(tenant.tenant_id),
                "role": user.role,
            }
        )

        return {
            "success": True,
            "data": {
                "token": token,
                "user_id": str(user.user_id),
                "tenant_id": str(tenant.tenant_id),
                "email": user.email,
                "role": user.role,
                "tenant_name": tenant.name,
            },
            "error": None,
        }

    except Exception as e:
        await db.rollback()
        return {
            "success": False,
            "data": None,
            "error": f"Registration failed: {str(e)}",
        }


@router.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT token."""
    try:
        # Find user by email
        result = await db.execute(
            select(User).where(User.email == request.email)
        )
        user = result.scalar_one_or_none()

        if user is None or not verify_password(request.password, user.hashed_password):
            return {
                "success": False,
                "data": None,
                "error": "Invalid email or password",
            }

        # Get tenant name
        tenant_result = await db.execute(
            select(Tenant).where(Tenant.tenant_id == user.tenant_id)
        )
        tenant = tenant_result.scalar_one()

        # Generate token
        token = create_access_token(
            data={
                "sub": user.email,
                "user_id": str(user.user_id),
                "tenant_id": str(user.tenant_id),
                "role": user.role,
            }
        )

        return {
            "success": True,
            "data": {
                "token": token,
                "user_id": str(user.user_id),
                "tenant_id": str(user.tenant_id),
                "email": user.email,
                "role": user.role,
                "tenant_name": tenant.name,
            },
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Login failed: {str(e)}",
        }


@router.get("/me")
async def get_me(user: CurrentUser = Depends(get_current_user)):
    """Return the current authenticated user's info."""
    return {
        "success": True,
        "data": {
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "role": user.role,
        },
        "error": None,
    }
