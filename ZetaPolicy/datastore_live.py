import numpy as np

from ZetaPolicy.constants import NUM_MFCC, NO_features, EMOTIONS
from ZetaPolicy.datastore import Datastore
from ZetaPolicy.feature_type import FeatureType


class LiveDatastore(Datastore):
    data_pkl = None
    data = []
    pre_train_data = []

    def __init__(self, feature_type: FeatureType) -> None:
        if not (FeatureType.MFCC == feature_type):
            raise RuntimeError("Only supports {}".format(FeatureType.MFCC.name))

        self.train_mfcc = np.empty([0, NUM_MFCC, NO_features])
        self.train_emotion = np.empty([0, len(EMOTIONS)])
        self.test_mfcc = np.empty([0, NUM_MFCC, NO_features])
        self.test_emotion = np.empty([0, len(EMOTIONS)])

        assert len(self.train_mfcc) == len(self.train_emotion)
        assert len(self.test_mfcc) == len(self.test_emotion)

    def get_data(self):
        return (self.train_mfcc, self.train_emotion, None), (None, None, None)

    def get_testing_data(self):
        return self.test_mfcc, self.test_emotion, None

    def add_data(self, mfcc, emotion):
        self.train_mfcc = np.append(self.train_mfcc, mfcc, axis=0)
        self.train_emotion = np.append(self.train_emotion, emotion, axis=0)
