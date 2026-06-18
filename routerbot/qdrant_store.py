import logging
import uuid as uuid_module
from embeddings_utils import embed_text
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Condition,
    PayloadSchemaType,
    FilterSelector,
)
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import QdrantClient
from config import QDRANT_COLLECTION_NAME
from qdrant_client_provider import get_qdrant_client

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

    ensure_payload_indexes()


def ensure_payload_indexes():
    client: QdrantClient = get_qdrant_client()

    indexed_fields = ["user_id", "doc_id"]

    for field_name in indexed_fields:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(
                f"qdrant_stage = payload_index_ensured field: {field_name} schema: keyword"
            )
        except Exception as error:
            logger.warning(
                f"qdrant_stage = payload_index_skip field: {field_name} reason: {str(error)}"
            )


def upsert_chunks(chunks: list[dict], user_id: str) -> dict:

    client: QdrantClient = get_qdrant_client()

    points = []

    for chunk in chunks:
        vector = embed_text(chunk["content"])

        point = PointStruct(
            id=str(uuid_module.uuid5(uuid_module.NAMESPACE_DNS, chunk["id"])),
            vector=vector,  # type: ignore
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


def retrieve_from_qdrant(
    query: str, user_id: str, top_k: int = 5, selected_doc_ids: list[str] | None = None
) -> list[dict]:
    client: QdrantClient = get_qdrant_client()

    query_vector = embed_text(query)

    must_conditions: list[Condition] = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]

    if selected_doc_ids:
        must_conditions.append(
            FieldCondition(key="doc_id", match=MatchAny(any=selected_doc_ids))
        )

    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=must_conditions),
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
        f"top_k: {top_k} returned: {len(chunks)} scoped: {bool(selected_doc_ids)}"
    )

    return chunks


def fetch_all_chunks_for_user(
    user_id: str, selected_doc_ids: list[str] | None = None
) -> list[dict]:
    client: QdrantClient = get_qdrant_client()

    must_conditions: list[Condition] = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]

    if selected_doc_ids:
        must_conditions.append(
            FieldCondition(key="doc_id", match=MatchAny(any=selected_doc_ids))
        )

    all_chunks = []
    next_page_offset = None

    while True:
        points, next_page_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(must=must_conditions),
            limit=100,
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            all_chunks.append(
                {
                    "id": payload.get("chunk_id", ""),
                    "title": payload.get("title", ""),
                    "content": payload.get("content", ""),
                }
            )

        if next_page_offset is None:
            break

    logger.info(
        f"qdrant_stage = fetch_all_done user_id: {user_id} "
        f"total_chunks: {len(all_chunks)} scoped: {bool(selected_doc_ids)}"
    )

    return all_chunks


def delete_chunks_by_doc_id(doc_id: str, user_id: str) -> dict:
    client: QdrantClient = get_qdrant_client()

    client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]
            )
        ),
    )

    logger.info(f"qdrant_stage = delete_done doc_id: {doc_id} user_id: {user_id}")

    return {"deleted": True, "doc_id": doc_id}
