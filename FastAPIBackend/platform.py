import logging
from os.path import exists

import tensorflow as tf
from keras import Model

from FastAPIBackend import memstore
from ZetaPolicy.constants import SL_MODEL_FILENAME, RL_MODEL_FILENAME


def initialize_platform():
    logging.log(logging.INFO, "Initializing Platform")

    memstore.SL_MODEL = _load_model(SL_MODEL_FILENAME)
    memstore.RL_MODEL = _load_model(RL_MODEL_FILENAME)
    logging.log(logging.INFO, "Platform Initialized")


def _load_model(model_filename: str) -> Model:
    model_file = f"./persistent_store/{model_filename}"

    if exists(model_file):
        return tf.keras.models.load_model(model_file)
    else:
        RuntimeError(f"{model_file} file not found!")
