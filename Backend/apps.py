from django.apps import AppConfig
from tensorflow.keras import Input

from Backend import rl
from ZetaPolicy.constants import NUM_MFCC, NO_features
from ZetaPolicy.data_versions import DataVersions
from ZetaPolicy.datastore_live import LiveDatastore
from ZetaPolicy.environments import LiveEnv
from ZetaPolicy.feature_type import FeatureType
from ZetaPolicy.models import get_model_9_rl
from ZetaPolicy.rl_custom_policy import ZetaPolicy
from rl_framework.agents import DQNAgent


class BackendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Backend'

    def ready(self):
        input_layer = Input(shape=(1, NUM_MFCC, NO_features))
        rl.MODEL = get_model_9_rl(input_layer, 'live')
        rl.DATASTORE = LiveDatastore(feature_type=FeatureType.MFCC)
        rl.ENV = LiveEnv(data_version=DataVersions.LIVE, datastore=rl.DATASTORE)
        rl.POLICY = ZetaPolicy(zeta_nb_steps=100000, eps=0.1)
        rl.AGENT = DQNAgent(model=rl.MODEL, nb_actions=rl.ENV.action_space.n, memory=rl.MEMORY, policy=rl.POLICY,
                            nb_steps_warmup=10, gamma=.99, target_model_update=100,
                            train_interval=4, delta_clip=1.,
                            enable_double_dqn=False,
                            enable_dueling_network=False,
                            dueling_type='avg')
        rl.AGENT.compile(optimizer='adam', metrics=['mae', 'accuracy'])

        print("Application READY!!!")
