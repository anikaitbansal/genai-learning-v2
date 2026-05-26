import logging
from qdrant_store import retrieve_from_qdrant
from config import RAG_TOP_K

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(self, user_id: str):
        self.user_id = user_id
        logger.info(f"QdrantRetriever initialized | user_id: {user_id}")

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        logger.info(
            f"retriever_stage = retrieve_start user_id: {self.user_id} "
            f"query_length: {len(query.strip())} top_k: {top_k}"
        )

        chunks = retrieve_from_qdrant(query, self.user_id, top_k)

        logger.info(
            f"retriever_stage = retrieve_done user_id: {self.user_id} "
            f"returned: {len(chunks)}"
        )

        return chunks
