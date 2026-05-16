import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from api_routes import router
from middleware import request_logging_middleware
from dependencies import initialize_retriever
from app_database import initialize_database

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    initialize_retriever()
    logger.info("Application startup completed successfully.")
    yield

app = FastAPI(
    title="AI Chatbot API",
    description="A FastAPI backend for an intent-based GenAI chatbot with memory, routing, and reset support.",
    version="1.0.0",
    lifespan=lifespan
)

app.middleware("http")(request_logging_middleware)

app.include_router(router)