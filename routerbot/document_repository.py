from uuid import UUID
from database import SessionLocal
from models import Document
import logging

logger = logging.getLogger(__name__)


class DocumentRepository:
    def save_document(self, filename: str, doc_id: str, user_id: UUID):
        db = SessionLocal()
        try:
            row = Document(user_id=user_id, filename=filename, doc_id=doc_id)
            db.add(row)
            db.commit()

            logger.info(
                f"document_stage = save_done user_id: {user_id} doc_id: {doc_id} filename: {filename}"
            )

        finally:
            db.close()

    def list_documents(self, user_id: UUID):
        db = SessionLocal()
        try:
            rows = (
                db.query(Document)
                .filter(Document.user_id == user_id)
                .order_by(Document.uploaded_at.desc())
                .all()
            )

            documents = [
                {
                    "doc_id": row.doc_id,
                    "filename": row.filename,
                    "uploaded_at": row.uploaded_at.isoformat(),
                }
                for row in rows
            ]

            logger.info(
                f"document_stage = list_done user_id: {user_id} count: {len(documents)}"
            )
            return documents

        finally:
            db.close()

    def delete_document(self, doc_id: str, user_id: UUID) -> bool:
        db = SessionLocal()
        try:
            deleted_count = (
                db.query(Document)
                .filter(Document.user_id == user_id, Document.doc_id == doc_id)
                .delete()
            )
            db.commit()

            found = deleted_count > 0

            logger.info(
                f"document_stage = delete_done user_id: {user_id} doc_id: {doc_id} found: {found}"
            )
            return found

        finally:
            db.close()
