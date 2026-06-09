"""
FastAPI dependencies — authentication and authorization.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token

# Bearer token extractor
bearer_scheme = HTTPBearer()


class CurrentUser:
    """Represents the authenticated user extracted from JWT."""

    def __init__(self, user_id: str, tenant_id: str, email: str, role: str):
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.email = email
        self.role = role


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    """
    Dependency that extracts and validates the JWT token from the
    Authorization header. Returns a CurrentUser object.
    """
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    tenant_id = payload.get("tenant_id")
    email = payload.get("sub")
    role = payload.get("role")

    if not all([user_id, tenant_id, email, role]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload incomplete",
        )

    return CurrentUser(
        user_id=user_id, tenant_id=tenant_id, email=email, role=role
    )


def require_role(*allowed_roles: str):
    """
    Dependency factory — restricts access to users with specific roles.

    Usage:
        @router.post("/admin-action", dependencies=[Depends(require_role("admin"))])
        async def admin_action(...): ...

    Or as a parameter dependency:
        async def endpoint(user: CurrentUser = Depends(require_role("admin", "writer"))):
    """

    async def role_checker(
        user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' not authorized. Required: {allowed_roles}",
            )
        return user

    return role_checker
