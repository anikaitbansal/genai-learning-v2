import logging
from datetime import datetime, timezone
from app_database import get_connection


logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, session_id):
        self.session_id = session_id



    def default_history(self):
        return [
            {"role": "system", "content": "You are a helpful assistant."}
        ]



    def load(self):
        default_history = self.default_history()

        logger.info(
            "memory_stage=load_start session_id=%s", 
            self.session_id)


        try:
            with get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("""
                    SELECT role, content
                    FROM session_history 
                    WHERE session_id = ? 
                    ORDER BY id ASC
                """, (self.session_id,))
                rows = cursor.fetchall()

            if not rows:
                logger.info(
                    "memory_stage=session_missing_using_default session_id=%s default_message_count=%s",
                    self.session_id,
                    len(default_history)
                )
                return default_history
            
            chat_history = []
            for row in rows:
                chat_history.append({
                    "role": row["role"],
                    "content": row["content"]
                })
            
            logger.info(
                "memory_stage=load_done session_id=%s message_count=%s",
                self.session_id,
                len(chat_history)
            )

            return chat_history
        
        except Exception as error:
            logger.warning(
                "memory_stage=load_failed session_id=%s error=%s",
                self.session_id,
                str(error)
            )

            logger.warning(
                "memory_stage=load_fallback_default session_id=%s default_message_count=%s",
                self.session_id,
                len(default_history)
            )

            return default_history
            


    def save(self, chat_history):
        logger.info(
            "memory_stage=save_start session_id=%s message_count=%s",
            self.session_id,
            len(chat_history)
        )

        with get_connection() as connection:
            cursor = connection.cursor()

            cursor.execute("""
                DELETE FROM session_history
                WHERE session_id = ?
            """, (self.session_id,))

            timestamp = datetime.now(timezone.utc).isoformat()

            for message in chat_history:
                cursor.execute("""
                    INSERT INTO session_history (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    self.session_id, 
                    message["role"], 
                    message["content"], 
                    timestamp))
                
            connection.commit()

        logger.info(
            "memory_stage=save_done session_id=%s message_count=%s",
            self.session_id,
            len(chat_history)
        )



    def clear(self):
        default_history = self.default_history()

        logger.info(
            "memory_stage=clear_start session_id=%s",
            self.session_id
        )

        with get_connection() as connection:
            cursor = connection.cursor()

            cursor.execute("""
                DELETE FROM session_history
                WHERE session_id = ?
            """, (self.session_id,))

            timestamp = datetime.now(timezone.utc).isoformat()

            for message in default_history:
                cursor.execute("""
                    INSERT INTO session_history (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    self.session_id,
                    message["role"],
                    message["content"],
                    timestamp
                ))

            connection.commit()

        logger.info(
            "memory_stage=clear_done session_id=%s message_count=%s",
            self.session_id,
            len(default_history)
        )

        return default_history