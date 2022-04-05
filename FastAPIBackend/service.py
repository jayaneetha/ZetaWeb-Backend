import logging
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
from fastapi import UploadFile
from sqlalchemy.orm import Session

from FastAPIBackend import memstore
from FastAPIBackend.db import crud
from FastAPIBackend.db.db_model import Feedback
from FastAPIBackend.db.schemas import FeedbackItemCreate
from ZetaPolicy.audio_utils import split_audio
from ZetaPolicy.constants import SR, DURATION, NUM_MFCC, NO_features, EMOTIONS

logger = logging.getLogger(__name__)


def process_file(file: UploadFile, db: Session):
    audio_id = str(uuid.uuid4())
    logger.info(f"audio id: {audio_id}")

    file_url = store_upload_file(file, audio_id)

    audio, sr = librosa.load(file_url, sr=SR)
    audio = split_audio(audio, sr, DURATION)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC)
    mfcc = deepcopy(mfcc)

    mfcc = mfcc.reshape(1, 1, NUM_MFCC, NO_features)
    rl_predictions = memstore.RL_MODEL.predict([mfcc])
    sl_predictions = memstore.SL_MODEL.predict([mfcc])
    rl_emotion = EMOTIONS[np.argmax(rl_predictions)]
    sl_emotion = EMOTIONS[np.argmax(sl_predictions)]

    feedback_item = FeedbackItemCreate(audio_id=audio_id, feedback=0.0, rl_emotion=rl_emotion, sl_emotion=sl_emotion)
    crud.create_feedback_item(db=db, feedback_item=feedback_item)

    memstore.STATE_STORE[audio_id] = mfcc

    return audio_id, rl_emotion, sl_emotion


def store_upload_file(file: UploadFile, filename: str):
    destination = Path(f"./media/{filename}{Path(file.filename).suffix}")
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    return destination


def add_feedback(audio_id: str, feedback: float, db: Session):
    feedback_item: Feedback = crud.get_item(db=db, audio_id=audio_id)
    feedback_item.feedback = feedback

    memstore.MEMORY.append(observation=memstore.STATE_STORE[audio_id],
                           action=feedback_item.rl_emotion, reward=feedback,
                           terminal=False, training=True)

    del memstore.STATE_STORE[audio_id]

    return crud.update_feedback_item(db=db, feedback_item=feedback_item)
