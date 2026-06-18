import logging
from collections import defaultdict
from qdrant_client_provider import get_qdrant_client
from config import QDRANT_COLLECTION_NAME
from database import SessionLocal
from models import Document
from qdrant_store import delete_chunks_by_doc_id

logger = logging.getLogger(__name__)


def reconcile_orphan_vectors(dry_run: bool = True) -> dict:
    logger.info(f"reconcile_stage = start dry_run: {dry_run}")

    db = SessionLocal()
    try:
        postgres_doc_ids = defaultdict(set)
        for row in db.query(Document.user_id, Document.doc_id).all():
            postgres_doc_ids[str(row.user_id)].add(row.doc_id)
    finally:
        db.close()

    client = get_qdrant_client()
    qdrant_doc_ids = defaultdict(set)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            qdrant_doc_ids[payload.get("user_id")].add(payload.get("doc_id"))
        if offset is None:
            break

    orphans_found = []
    deleted_count = 0

    for user_id in set(postgres_doc_ids) | set(qdrant_doc_ids):
        orphans = qdrant_doc_ids[user_id] - postgres_doc_ids[user_id]
        for doc_id in orphans:
            orphans_found.append({"user_id": user_id, "doc_id": doc_id})
            if not dry_run:
                delete_chunks_by_doc_id(doc_id, user_id)
                deleted_count += 1

    logger.info(
        f"reconcile_stage = done dry_run: {dry_run} "
        f"orphans_found: {len(orphans_found)} deleted: {deleted_count}"
    )

    return {
        "dry_run": dry_run,
        "orphans_found": len(orphans_found),
        "deleted": deleted_count,
        "orphans": orphans_found,
    }
