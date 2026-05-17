from datetime import datetime, timezone
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Session as SessionModel, SessionHistory
import logging

logger = logging.getLogger(__name__)


def get_or_create_session_id(db: Session, session_key: str) -> int:
    session = db.query(SessionModel).filter(SessionModel.session_key == session_key).first()

    if session is None:
        session = SessionModel(user_id=1, session_key=session_key)
        db.add(session)
        db.commit()
        db.refresh(session)

    return session.id



class MemoryManager:
    def __init__(self, session_id: str):
        self.session_id = session_id



    def default_history(self):
        return [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    


    def load(self):
        default_history = self.default_history()
        logger.info(f"memory_stage = load_start Session: {self.session_id}")

        try:
            db = SessionLocal()
            try:
                session_pk = get_or_create_session_id(db, self.session_id)

                rows = (
                    db.query(SessionHistory)
                    .filter(SessionHistory.session_id == session_pk)
                    .order_by(SessionHistory.id.asc())
                    .all()
                )

                if not rows:
                    logger.info(f"memory_stage = session_missing_using_default Session: {self.session_id}")
                    return default_history

                chat_history = [{"role": row.role, "content": row.content} for row in rows]

                logger.info(f"memory_stage = load_done Session: {self.session_id} message_count: {len(chat_history)}")
                return chat_history

            finally:
                db.close()

        except Exception as error:
            logger.warning(f"memory_stage = load_failed Session: {self.session_id} error: {str(error)}")
            return default_history
        


    def save(self, chat_history):
        logger.info(f"memory_stage = save_start Session: {self.session_id} message_count: {len(chat_history)}")

        db = SessionLocal()
        try:
            session_pk = get_or_create_session_id(db, self.session_id)

            db.query(SessionHistory).filter(SessionHistory.session_id == session_pk).delete()

            timestamp = datetime.now(timezone.utc)

            for message in chat_history:
                row = SessionHistory(
                    session_id=session_pk,
                    role=message["role"],
                    content=message["content"],
                    created_at=timestamp
                )
                db.add(row)

            db.commit()

            logger.info(f"memory_stage = save_done Session: {self.session_id} message_count: {len(chat_history)}")

        finally:
            db.close()



    def clear(self):
        default_history = self.default_history()
        logger.info(f"memory_stage = clear_start Session: {self.session_id}")

        db = SessionLocal()
        try:
            session_pk = get_or_create_session_id(db, self.session_id)

            db.query(SessionHistory).filter(SessionHistory.session_id == session_pk).delete()

            timestamp = datetime.now(timezone.utc)

            for message in default_history:
                row = SessionHistory(
                    session_id=session_pk,
                    role=message["role"],
                    content=message["content"],
                    created_at=timestamp
                )
                db.add(row)

            db.commit()

            logger.info(f"memory_stage = clear_done Session: {self.session_id} message_count: {len(default_history)}")
            return default_history

        finally:
            db.close()
    