from app_database import get_connection


class ChatLogRepository:
    def save_chat_log(self, session_id, request_id, user_message, bot_reply, intent, rag_used):
        with get_connection() as connection:
            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO chat_logs (
                    session_id,
                    request_id,
                    user_message,
                    bot_reply,
                    intent,
                    rag_used
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                request_id,
                user_message,
                bot_reply,
                intent,
                int(rag_used)
            ))

            connection.commit()