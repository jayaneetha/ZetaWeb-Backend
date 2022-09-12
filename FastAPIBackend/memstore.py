import numpy as np
from keras import Model

from ZetaPolicy.constants import WINDOW_LENGTH
from ZetaPolicy.datastore_live import LiveDatastore
from ZetaPolicy.environments import AbstractEnv
from rl.core import Agent
from rl.memory import SequentialMemory
from rl.policy import Policy

RL_MODEL: Model = None
ENV: AbstractEnv = None
DATASTORE: LiveDatastore = None
STATE_STORE = {}
MEMORY = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
POLICY: Policy = None
AGENT: Agent = None

SL_MODEL: Model

step = np.int16(0)
