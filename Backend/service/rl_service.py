import logging
import uuid
from copy import deepcopy

import librosa
import numpy as np
import tensorflow as tf
from django.core.files.storage import default_storage

from Backend import rl
from Backend.models import State
from Backend.rl import MODEL, STATE_STORE
from ZetaPolicy.audio_utils import split_audio
from ZetaPolicy.constants import SR, DURATION, NUM_MFCC, NO_features, EMOTIONS


def store_file(file):
    file_name = default_storage.save(file.name, file)
    file_url = default_storage.path(file_name)
    return file_url


def process_audio(file_url):
    print("Tensorflow version:", tf.__version__)

    audio_id = str(uuid.uuid1())

    audio, sr = librosa.load(file_url, sr=SR)
    audio = split_audio(audio, sr, DURATION)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC)

    STATE_STORE[audio_id] = mfcc

    # ds = IemocapDatastore(FeatureType.MFCC)
    # (_mfcc, _, _), (_) = ds.get_data()
    #
    # env = IemocapEnv(DataVersions.IEMOCAP, datastore=ds)
    #
    # nb_actions = env.action_space.n
    #
    # input_layer = Input(shape=(1, NUM_MFCC, NO_features))
    #
    # model = models.get_model_9_rl(input_layer, model_name_prefix='mfcc')
    #
    # memory = SequentialMemory(limit=1000000, window_length=1)
    #
    # policy = ZetaPolicy(zeta_nb_steps=5000, eps=0.1)
    #
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
    #                nb_steps_warmup=50, gamma=.99, target_model_update=10000,
    #                train_interval=4, delta_clip=1.)
    # dqn.compile('adam', metrics=['mae', 'accuracy'])
    # dqn.fit(env, callbacks=[], nb_steps=500, log_interval=10)
    mfcc = deepcopy(mfcc)
    rl.AGENT.reset_states()
    # action = rl.AGENT.forward(mfcc)

    mfcc = mfcc.reshape(1, 1, NUM_MFCC, NO_features)
    predictions = MODEL.predict([mfcc])
    emotion = EMOTIONS[np.argmax(predictions)]

    state = State()
    state.uuid = audio_id
    state.emotion = emotion
    state.save()

    logging.log(logging.INFO, [audio_id, predictions, EMOTIONS[np.argmax(predictions)]])
    return audio_id, emotion


def process_feedback(audio_id, feedback):
    logging.log(logging.INFO, [audio_id, feedback])
    s = State.objects.get(uuid=audio_id)
    reward = reward_function(feedback)
    s.reward = reward
    s.save()


def reward_function(feedback):
    feedback = int(feedback)
    if feedback == 1:
        return 0.9
    elif feedback == 0:
        return -0.9
    else:
        raise RuntimeError("invalid feedback")
