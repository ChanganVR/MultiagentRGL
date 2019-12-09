from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'rgl_ppo'

        # gcn
        self.gcn.num_layer = 2
        self.gcn.X_dim = 32
        self.gcn.similarity_function = 'embedded_gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = True

        self.rgl_ppo = Config()
        self.rgl_ppo.motion_predictor_dims = [64, 5]
        self.rgl_ppo.value_network_dims = [32, 100, 100, 1]


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)

        self.imitation_learning.il_epochs = 50
        self.train.rl_train_epochs = 1
        self.train.rl_learning_rate = 2.5e-4
        self.train.train_episodes = 1e6

        if debug:
            self.imitation_learning.il_episodes = 10
            self.imitation_learning.il_epochs = 5
            self.train.train_episodes = 1
            self.train.checkpoint_interval = self.train.train_episodes
            self.train.evaluation_interval = 1
            self.train.target_update_interval = 1
