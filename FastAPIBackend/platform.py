import logging
from os.path import exists

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model

from FastAPIBackend import memstore
from ZetaPolicy.constants import NUM_MFCC, NO_features, SL_MODEL_FILENAME
from ZetaPolicy.data_versions import DataVersions
from ZetaPolicy.datastore_live import LiveDatastore
from ZetaPolicy.environments import LiveEnv
from ZetaPolicy.feature_type import FeatureType
from ZetaPolicy.models import get_model_9_rl
from ZetaPolicy.rl_custom_policy import ZetaPolicy
from rl_framework.rl2.agents import DQNAgent


def initialize_platform():
    logging.log(logging.INFO, "Initializing Platform")
    input_layer = Input(shape=(1, NUM_MFCC, NO_features))
    memstore.RL_MODEL = get_model_9_rl(input_layer, 'live')
    memstore.DATASTORE = LiveDatastore(feature_type=FeatureType.MFCC)
    memstore.ENV = LiveEnv(data_version=DataVersions.LIVE, datastore=memstore.DATASTORE)
    memstore.POLICY = ZetaPolicy(zeta_nb_steps=100000, eps=0.1)
    memstore.AGENT = DQNAgent(model=memstore.RL_MODEL, nb_actions=memstore.ENV.action_space.n, memory=memstore.MEMORY,
                              policy=memstore.POLICY,
                              batch_size=4,
                              nb_steps_warmup=15, gamma=.99, target_model_update=100,
                              train_interval=4, delta_clip=1.,
                              enable_double_dqn=False,
                              enable_dueling_network=False,
                              dueling_type='avg')
    memstore.AGENT.compile(optimizer='adam', metrics=['mae', 'accuracy'])
    memstore.AGENT.set_training(training=True)

    memstore.SL_MODEL = _load_sl_model()
    logging.log(logging.INFO, "Platform Initialized")


def _load_sl_model() -> Model:
    model_file = f"./persistent_store/{SL_MODEL_FILENAME}"

    if exists(model_file):
        return tf.keras.models.load_model(model_file)
    else:
        RuntimeError(f"{model_file} file not found!")
