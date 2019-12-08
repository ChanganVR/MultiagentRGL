from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.sim.human_num = 2


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

        self.train.rl_train_epochs = 1
        self.train.rl_learning_rate = 2.5e-4
        self.train.train_episodes = 1e6
