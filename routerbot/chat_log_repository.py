from database import SessionLocal
from models import ChatLog
from memory_manager import get_or_create_session_id
import logging

logger = logging.getLogger(__name__)


class ChatLogRepository:
    def save_chat_log(self, session_id, request_id, user_message, bot_reply, intent, rag_used):
        db = SessionLocal()
        try:
            session_pk = get_or_create_session_id(db, session_id)

            row = ChatLog(
                user_id=1,
                session_id=session_pk,
                request_id=request_id,
                user_input=user_message,
                bot_response=bot_reply,
                intent=intent,
                rag_used=rag_used
            )
            db.add(row)
            db.commit()

            logger.info(f"chat_log_stage = save_done session_id: {session_id} request_id: {request_id} intent: {intent} rag_used: {rag_used}")

        finally:
            db.close()