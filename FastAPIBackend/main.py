import logging
from typing import List

from fastapi import FastAPI, Depends
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from FastAPIBackend import service
from FastAPIBackend.db import db_model, crud, schemas
from FastAPIBackend.db.database import engine, SessionLocal
from FastAPIBackend.platform import initialize_platform

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

logger = logging.getLogger(__name__)

db_model.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

initialize_platform()


@app.post("/uploadfile")
def create_upload_file(audio_file: UploadFile, db: Session = Depends(get_db)):
    audio_id, rl_emotion, sl_emotion = service.process_file(audio_file, db)
    return {'audio_id': audio_id, 'rl_emotion': rl_emotion, 'sl_emotion': sl_emotion}

@app.post("/feedback")
def add_feedback(audio_id: str, feedback: bool, model: str, db: Session = Depends(get_db)):
    feedback_item = service.add_feedback(audio_id, model, feedback, db)
    return {'feedback': feedback_item}


@app.get("/items/", response_model=List[schemas.Feedback])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items


@app.get("/performance")
def get_model_performance(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return service.get_model_performance(skip, limit, db)
