from datetime import datetime, timezone
from sqlalchemy import func
from database import SessionLocal
from models import Feedback
from memory_manager import get_or_create_session_id
import logging

logger = logging.getLogger(__name__)


class FeedbackManager:
    def create_feedback_entry(self, session_id, request_id, rating, comments=None):
        return {
            "session_id": session_id,
            "request_id": request_id,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    


    def save_feedback(self, feedback_entry):
        db = SessionLocal()
        try:
            session_pk = get_or_create_session_id(db, feedback_entry["session_id"])

            row = Feedback(
                user_id=1,
                session_id=session_pk,
                request_id=feedback_entry["request_id"],
                rating=feedback_entry["rating"],
                comments=feedback_entry.get("comments"),
                timestamp=datetime.now(timezone.utc)
            )
            db.add(row)
            db.commit()
        finally:
            db.close()

            

    def get_summary(self):
        db = SessionLocal()
        try:
            total_feedback = db.query(func.count(Feedback.id)).scalar()

            if total_feedback == 0:
                return {
                    "total_feedback": 0,
                    "average_rating": 0,
                    "ratings_count": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
                }

            average_rating = round(db.query(func.avg(Feedback.rating)).scalar(), 2)

            rows = db.query(Feedback.rating, func.count(Feedback.id)).group_by(Feedback.rating).all()

            ratings_count = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
            for rating, count in rows:
                ratings_count[str(rating)] = count

            return {
                "total_feedback": total_feedback,
                "average_rating": average_rating,
                "ratings_count": ratings_count
            }
        finally:
            db.close()