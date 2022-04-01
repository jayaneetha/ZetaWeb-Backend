from keras import Model

from ZetaPolicy.constants import WINDOW_LENGTH
from ZetaPolicy.datastore_live import LiveDatastore
from ZetaPolicy.environments import AbstractEnv
from rl_framework.rl1.core import Agent
from rl_framework.rl1.memory import SequentialMemory
from rl_framework.rl1.policy import Policy

MODEL: Model = None
ENV: AbstractEnv = None
DATASTORE: LiveDatastore = None
STATE_STORE = {}
MEMORY = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
POLICY: Policy = None
AGENT: Agent = None
