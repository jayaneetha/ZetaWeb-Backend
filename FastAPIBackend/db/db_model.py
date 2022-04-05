from sqlalchemy import Column, Integer, String, Float

from FastAPIBackend.db.database import Base


class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String, index=True)
    feedback = Column(Float, default=0.0)
    rl_emotion = Column(String)
    sl_emotion = Column(String)
