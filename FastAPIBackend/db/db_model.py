from sqlalchemy import Column, Integer, String, Boolean, Float

from FastAPIBackend.db.database import Base


class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String, index=True)
    original_filename = Column(String)
    rl_feedback = Column(Boolean, nullable=True)
    sl_feedback = Column(Boolean, nullable=True)
    rl_emotion = Column(String)
    sl_emotion = Column(String)


class Accuracies(Base):
    __tablename__ = "accuracies"
    episode = Column(Integer, primary_key=True, index=True)
    rl_accuracy = Column(Float)
    sl_accuracy = Column(Float)
