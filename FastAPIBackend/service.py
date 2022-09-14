import logging
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from FastAPIBackend import memstore
from FastAPIBackend.db import crud
from FastAPIBackend.db.db_model import Feedback
from FastAPIBackend.db.schemas import FeedbackItemCreate
from ZetaPolicy.audio_utils import split_audio
from ZetaPolicy.constants import SR, DURATION, NUM_MFCC, NO_features, EMOTIONS

logger = logging.getLogger(__name__)


def process_file(file: UploadFile, db: Session):
    logger.info(file.content_type)

    if file.content_type not in ['audio/wave', 'audio/wav']:
        raise HTTPException(status_code=400, detail=f"Invalid file type {file.content_type}. Required: audio/wav")

    audio_id = str(uuid.uuid4())
    logger.info(f"audio id: {audio_id}")

    file_url = store_upload_file(file, audio_id)

    audio, sr = librosa.load(file_url, sr=SR)
    audio_frames = split_audio(audio, sr, DURATION)
    if len(audio_frames) > 0:
        audio = audio_frames[0]
    else:
        raise HTTPException(status_code=400, detail=f"Audio Length is smaller than {DURATION}s")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC)
    mfcc = deepcopy(mfcc)

    mfcc = mfcc.reshape(1, 1, NUM_MFCC, NO_features)

    rl_predictions = memstore.RL_MODEL.predict([mfcc], verbose=False)
    sl_predictions = memstore.SL_MODEL.predict([mfcc], verbose=False)

    rl_emotion = EMOTIONS[np.argmax(rl_predictions)]
    sl_emotion = EMOTIONS[np.argmax(sl_predictions)]

    feedback_item = FeedbackItemCreate(audio_id=audio_id, rl_emotion=rl_emotion, sl_emotion=sl_emotion,
                                       original_filename=file.filename, processed=False)
    crud.create_feedback_item(db=db, feedback_item=feedback_item)

    return audio_id, rl_emotion, sl_emotion


def store_upload_file(file: UploadFile, filename: str):
    destination = Path(f"./media/{filename}{Path(file.filename).suffix}")
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    return destination


def add_feedback(audio_id: str, model: str, feedback: bool, db: Session):
    model = model.upper()
    if model == 'RL' or model == 'SL':
        feedback_item: Feedback = crud.get_item(db=db, audio_id=audio_id)

        if feedback_item is not None:
            if model == 'RL':
                feedback_item.rl_feedback = feedback

            if model == 'SL':
                feedback_item.sl_feedback = feedback

            return crud.update_feedback_item(db=db, feedback_item=feedback_item)
        else:
            logger.error(f"Invalid audio id value: {audio_id}")
            raise HTTPException(status_code=404, detail=f"audio id not found: {audio_id}")
    else:
        logger.error(f"Invalid model value: {model}")
        raise HTTPException(status_code=400, detail=f"Invalid model value: {model}")


def get_model_performance(skip: int, limit: int, db: Session) -> list:
    acc = crud.get_accuracies(db, skip, limit)
    return acc
