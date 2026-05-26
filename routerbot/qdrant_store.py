import logging
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import QdrantClient
from config import QDRANT_COLLECTION_NAME
from qdrant_client_provider import get_qdrant_client
from embeddings_utils import embed_text
import uuid as uuid_module

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 384


def ensure_collection_exists():
    client: QdrantClient = get_qdrant_client()

    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
        logger.info(f"qdrant_stage = collection_exists name: {QDRANT_COLLECTION_NAME}")

    except UnexpectedResponse:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION, distance=Distance.COSINE
            ),
        )
        logger.info(
            f"qdrant_stage = collection_created name: {QDRANT_COLLECTION_NAME} dimension: {EMBEDDING_DIMENSION}"
        )


def upsert_chunks(chunks: list[dict], user_id: str) -> dict:
    client: QdrantClient = get_qdrant_client()

    points = []

    for chunk in chunks:
        vector = embed_text(chunk["content"])

        point = PointStruct(
            id=str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk["id"])),
            vector=vector,
            payload={
                "user_id": user_id,
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["id"],
                "title": chunk["title"],
                "content": chunk["content"],
            },
        )
        points.append(point)

    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    logger.info(
        f"qdrant_stage = upsert_done collection: {QDRANT_COLLECTION_NAME} "
        f"user_id: {user_id} chunk_count: {len(points)}"
    )

    return {"added_chunks": len(points)}


def retrieve_from_qdrant(query: str, user_id: str, top_k: int = 5) -> list[dict]:
    client: QdrantClient = get_qdrant_client()

    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            {
                "id": payload.get("chunk_id", ""),
                "title": payload.get("title", ""),
                "content": payload.get("content", ""),
                "score": round(point.score, 4),
            }
        )

    logger.info(
        f"qdrant_stage = retrieve_done user_id: {user_id} "
        f"top_k: {top_k} returned: {len(chunks)}"
    )

    return chunks
