import uuid
import os
import logging
from typing import Optional
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, exceptions
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.jwt import decode_jwt, generate_jwt
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from models import User
from token_blocklist import is_blocklisted

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
ASYNC_DATABASE_URL = DATABASE_URL.replace(
    "postgresql+psycopg2://", "postgresql+asyncpg://"
).replace("postgresql://", "postgresql+asyncpg://")

SECRET = os.environ.get("AUTH_SECRET", "CHANGE_ME_IN_PRODUCTION")

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_db)):
    yield SQLAlchemyUserDatabase(session, User)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"auth_stage = user_registered user_id: {user.id}")

    async def on_after_login(
        self, user: User, request: Optional[Request] = None, response=None
    ):
        logger.info(f"auth_stage = user_login user_id: {user.id}")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"auth_stage = forgot_password user_id: {user.id}")

    async def on_after_reset_password(
        self, user: User, request: Optional[Request] = None
    ):
        logger.info(f"auth_stage = password_reset user_id: {user.id}")


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


bearer_transport = BearerTransport(tokenUrl="auth/login")


class RedisAwareJWTStrategy(JWTStrategy):
    async def write_token(self, user: User) -> str:
        jti = str(uuid.uuid4())
        data = {
            "sub": str(user.id),
            "aud": self.token_audience,
            "jti": jti,
        }
        token = generate_jwt(
            data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm
        )
        logger.info(f"auth_stage = token_issued user_id: {user.id} jti: {jti}")
        return token

    async def read_token(
        self, token: Optional[str], user_manager: BaseUserManager
    ) -> Optional[User]:
        if token is None:
            return None

        try:
            data = decode_jwt(
                token, self.decode_key, self.token_audience, algorithms=[self.algorithm]
            )
        except Exception:
            logger.info("auth_stage = token_rejected reason = decode_failed")
            return None

        jti = data.get("jti")
        if jti and is_blocklisted(jti):
            logger.info(f"auth_stage = token_rejected reason = blocklisted jti: {jti}")
            return None

        user_id = data.get("sub")
        if user_id is None:
            return None

        try:
            parsed_user_id = uuid.UUID(user_id)
            return await user_manager.get(parsed_user_id)
        except (exceptions.UserNotExists, ValueError):
            return None


def get_jwt_strategy() -> JWTStrategy:
    return RedisAwareJWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)
