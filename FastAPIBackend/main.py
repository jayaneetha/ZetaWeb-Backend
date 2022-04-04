import logging

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

from FastAPIBackend.API_Platform import initialize_platform
from FastAPIBackend.service import process_file

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

initialize_platform()


@app.post("/uploadfile")
def create_upload_file(audioFile: UploadFile):
    audio_id, rl_emotion, sl_emotion = process_file(audioFile)
    return {'audio_id': audio_id, 'rl_emotion': rl_emotion, 'sl_emotion': sl_emotion}
