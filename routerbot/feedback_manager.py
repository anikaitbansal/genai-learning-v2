from app_database import get_connection
from datetime import datetime, timezone

class FeedbackManager:

    def create_feedback_entry(self, session_id, request_id, rating, comments="None"):
        return {
            "session_id": session_id,
            "request_id": request_id,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


      
    def save_feedback(self, feedback_entry):
        with get_connection() as connection:
            cursor = connection.cursor()

            cursor.execute("""
                INSERT INTO feedback (session_id, request_id, rating, comments, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                feedback_entry["session_id"],
                feedback_entry["request_id"],
                feedback_entry["rating"],
                feedback_entry["comments"],
                feedback_entry["timestamp"]
            ))

            connection.commit()



    def get_summary(self):
        with get_connection() as connection:
            cursor = connection.cursor()

            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]

            if total_feedback == 0:
                return {
                    "total_feedback": 0,
                    "average_rating": 0,
                    "rating_counts": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
                }

            cursor.execute("SELECT AVG(rating) FROM feedback")
            average_rating = round(cursor.fetchone()[0], 2)

            cursor.execute("""
                SELECT rating, COUNT(*) 
                FROM feedback 
                GROUP BY rating
            """)
            rows = cursor.fetchall()

            rating_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

            for row in rows:
                rating_counts[str(row[0])] = row[1]

            return {
                "total_feedback": total_feedback,
                "average_rating": average_rating,
                "rating_counts": rating_counts
            }