import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from api_routes import router
from middleware import request_logging_middleware
from dependencies import initialize_retriever
from app_database import initialize_database
from auth import fastapi_users, auth_backend
from schemas import UserRead, UserCreate, UserUpdate
from database import verify_connection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    verify_connection()
    initialize_retriever()
    logger.info("Application startup completed successfully.")
    yield


app = FastAPI(
    title="AI Chatbot API",
    description="A FastAPI backend for an intent-based GenAI chatbot with memory, routing, and reset support.",
    version="2.0.0",
    lifespan=lifespan,
)

app.middleware("http")(request_logging_middleware)

app.include_router(router)

app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth", tags=["Auth"]
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["Users"],
)
