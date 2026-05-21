import time
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi_users.jwt import decode_jwt
from auth import current_active_user, SECRET, bearer_transport
from models import User
from token_blocklist import add_to_blocklist

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/logout")
async def logout(
    http_request: Request,
    user: User = Depends(current_active_user),
    token: str = Depends(bearer_transport.scheme),
):
    request_id = http_request.state.request_id

    try:
        data = decode_jwt(
            token,
            SECRET,
            audience=["fastapi-users:auth"],
            algorithms=["HS256"],
        )
    except Exception:
        logger.warning(f"Request ID: {request_id} - endpoint = /auth/logout stage = decode_failed user_id: {user.id}")
        raise HTTPException(status_code=401, detail="Invalid token")

    jti = data.get("jti")
    exp = data.get("exp")

    if not jti or not exp:
        logger.warning(f"Request ID: {request_id} - endpoint = /auth/logout stage = missing_claims user_id: {user.id}")
        raise HTTPException(status_code=400, detail="Token missing required claims")

    remaining_ttl = int(exp - time.time())

    add_to_blocklist(jti, remaining_ttl)

    logger.info(
        f"Request ID: {request_id} - endpoint = /auth/logout stage = logout_done "
        f"user_id: {user.id} jti: {jti} ttl_seconds: {remaining_ttl}"
    )

    return {"message": "Logged out successfully"}