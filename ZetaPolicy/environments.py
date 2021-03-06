import gym
import numpy as np

from ZetaPolicy.constants import EMOTIONS, NUM_MFCC, NO_features
from ZetaPolicy.datastore import CombinedDatastore
from ZetaPolicy.datastore_esd import ESDDatastore
from ZetaPolicy.datastore_iemocap import IemocapDatastore
from ZetaPolicy.datastore_improv import ImprovDatastore
from ZetaPolicy.datastore_live import LiveDatastore
from ZetaPolicy.datastore_savee import SaveeDatastore
from ZetaPolicy.feature_type import FeatureType


class AbstractEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore) -> None:
        super().__init__()
        self.itr = 0

        self.X = []
        self.Y = []
        self.num_classes = len(EMOTIONS)

        self.data_version = data_version

        self.datastore = datastore

        self.set_data()

        self.action_space = gym.spaces.Discrete(self.num_classes)
        self.observation_space = gym.spaces.Box(-1, 1, [NUM_MFCC, NO_features])

    def step(self, action):
        assert self.action_space.contains(action)
        reward = -0.1 + int(action == np.argmax(self.Y[self.itr]))
        # reward = 1 if action == self.Y[self.itr] else -1

        done = (len(self.X) - 2 <= self.itr)

        next_state = self.X[self.itr + 1]

        info = {
            "ground_truth": np.argmax(self.Y[self.itr]),
            "itr": self.itr,
            "correct_inference": int(action == np.argmax(self.Y[self.itr]))
        }
        self.itr += 1

        return next_state, reward, done, info

    def render(self, mode='human'):
        print("Not implemented \t i: {}".format(self.itr))

    def reset(self):
        self.itr = 0
        self.set_data()
        return self.X[self.itr]

    def set_data(self):
        self.X = []
        self.Y = []

        (x_train, y_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = self.datastore.get_data()

        assert len(x_train) == len(y_train)
        self.X = x_train
        self.Y = y_train


class IemocapEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: IemocapDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = IemocapDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class ImprovEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: ImprovDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = ImprovDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class SaveeEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version) -> None:
        super().__init__(data_version=data_version, datastore=SaveeDatastore(FeatureType.MFCC))


class ESDEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: ESDDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = ESDDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class CombinedEnv(AbstractEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: CombinedDatastore = None, custom_split: float = None) -> None:
        if datastore is None:
            datastore = CombinedDatastore(FeatureType.MFCC, custom_split)
        super().__init__(data_version=data_version, datastore=datastore)


class LiveEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version, datastore: LiveDatastore) -> None:
        super().__init__()
        if datastore is None:
            datastore = LiveDatastore(FeatureType.MFCC)

        self.itr = 0

        self.num_classes = len(EMOTIONS)
        self.action_space = gym.spaces.Discrete(self.num_classes)
        self.observation_space = gym.spaces.Box(-1, 1, [NUM_MFCC, NO_features])
