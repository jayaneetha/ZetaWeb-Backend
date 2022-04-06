from pydantic import BaseModel


class FeedbackItemBase(BaseModel):
    audio_id: str
    original_filename: str
    feedback: float
    rl_emotion: str
    sl_emotion: str


class FeedbackItemCreate(FeedbackItemBase):
    pass


class Feedback(FeedbackItemBase):
    id: int

    class Config:
        orm_mode = True
