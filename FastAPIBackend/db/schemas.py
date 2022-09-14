from typing import Optional

from pydantic import BaseModel


class FeedbackItemBase(BaseModel):
    audio_id: str
    original_filename: str
    rl_feedback: Optional[bool]
    sl_feedback: Optional[bool]
    processed: Optional[bool]
    rl_emotion: str
    sl_emotion: str


class FeedbackItemCreate(FeedbackItemBase):
    pass


class Feedback(FeedbackItemBase):
    id: int

    class Config:
        orm_mode = True
