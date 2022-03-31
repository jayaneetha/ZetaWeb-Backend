from Backend import sl
from ZetaPolicy.constants import NUM_MFCC, NO_features
from ZetaPolicy.datastore_esd import ESDDatastore
from ZetaPolicy.feature_type import FeatureType
from ZetaPolicy.framework import train


def train_sl(epochs=64, batch_size=8):
    ds = ESDDatastore(FeatureType.MFCC, custom_split=0.8)
    (x_train, y_train, _), _ = ds.get_data()

    history, trained_model = train(model=sl.MODEL, x=x_train.reshape((len(x_train), 1, NUM_MFCC, NO_features)),
                                   y=y_train, epochs=epochs, batch_size=batch_size)
    sl.MODEL = trained_model
    pass
