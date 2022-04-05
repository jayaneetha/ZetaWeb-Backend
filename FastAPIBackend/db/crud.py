from sqlalchemy.orm import Session

from FastAPIBackend.db import db_model, schemas
from FastAPIBackend.db.db_model import Feedback


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(db_model.Feedback).offset(skip).limit(limit).all()


def get_item(db: Session, audio_id: str):
    return db.query(db_model.Feedback).filter(db_model.Feedback.audio_id == audio_id).first()


def create_feedback_item(db: Session, feedback_item: schemas.FeedbackItemCreate):
    db_item = db_model.Feedback(**feedback_item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def update_feedback_item(db: Session, feedback_item: Feedback):
    db.add(feedback_item)
    db.commit()
    db.refresh(feedback_item)
    return feedback_item
