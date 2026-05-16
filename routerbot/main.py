import logging
from fastapi import FastAPI
from api_routes import router
from middleware import request_logging_middleware
from dependencies import initialize_retriever
from app_database import initialize_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = logging.getLogger(__name__)


app = FastAPI(
    title = "GenAI RouterBot API",
    description = " A FastAPI backend for an intent-based GenAI chatbot with memory, routing and reset support.",
    version = "1.0.0"
)

@app.on_event("startup")
def startup_event():
    initialize_database()
    initialize_retriever()
    logger.info("Application startup completed successfully.")

app.middleware("http")(request_logging_middleware)

app.include_router(router)