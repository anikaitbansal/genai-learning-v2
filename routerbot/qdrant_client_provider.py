import logging
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY

logger = logging.getLogger(__name__)

_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        logger.info("qdrant_stage = client_created")

    return _qdrant_client


def verify_qdrant_connection():
    try:
        client = get_qdrant_client()
        client.get_collections()
        logger.info("qdrant_stage = connection_verified")
    except Exception as error:
        logger.error(f"qdrant_stage = connection_failed error: {str(error)}")
        raise
