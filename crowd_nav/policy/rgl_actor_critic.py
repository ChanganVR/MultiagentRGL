import numpy as np
import torch

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.value_action_predictor import ValueActionPredictor
from torch.distributions.multivariate_normal import MultivariateNormal


class RglActorCritic(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'rgl_ppo'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.value_action_predictor = None
        self.traj = None
        self.shared_gcn = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.shared_gcn = True
        self.value_action_predictor = ValueActionPredictor(config, self.robot_state_dim, self.human_state_dim,
                                                           shared_gcn=self.shared_gcn)
        if self.shared_gcn:
            self.model = [self.value_action_predictor.graph_model, self.value_action_predictor.value_network,
                          self.value_action_predictor.action_network]
        else:
            self.model = [self.value_action_predictor.graph_model_val, self.value_action_predictor.graph_model_act,
                          self.value_action_predictor.value_network, self.value_action_predictor.action_network]

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_time_step(self, time_step):
        self.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_action_predictor

    def get_state_dict(self):
        if self.shared_gcn:
            return {
                'graph_model': self.value_action_predictor.graph_model.state_dict(),
                'value_network': self.value_action_predictor.value_network.state_dict(),
                'action_network': self.value_action_predictor.action_network.state_dict()
            }
        else:
            return {
                'graph_model_val': self.value_action_predictor.graph_model_val.state_dict(),
                'graph_model_act': self.value_action_predictor.graph_model_act.state_dict(),
                'value_network': self.value_action_predictor.value_network.state_dict(),
                'action_network': self.value_action_predictor.action_network.state_dict()
            }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        if self.shared_gcn:
            self.value_action_predictor.graph_model.load_state_dict(state_dict['graph_model'])
        else:
            self.value_action_predictor.graph_model_val.load_state_dict(state_dict['graph_model_val'])
            self.value_action_predictor.graph_model_act.load_state_dict(state_dict['graph_model_act'])
        self.value_action_predictor.value_network.load_state_dict(state_dict['value_network'])
        self.value_action_predictor.action_network.load_state_dict(state_dict['action_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def act(self, state_tensor): # state is a batch of tensors rather than a joint state
        value, mu, cov = self.value_action_predictor(state_tensor)
        dist = MultivariateNormal(mu, cov)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        action_to_take = [ActionXY(action[0], action[1]) for action in actions.cpu().numpy()]
        
        return value, actions, action_log_probs, action_to_take
    
    def predict(self, state): # used in Agent class; state must be a joint state
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        # if self.reach_destination(state):
        #     return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
        value, action, action_log_probs, action_to_take = self.act(state_tensor)
        if self.phase == 'train':
            self.last_state = self.transform(state)
            
        return value[0], action[0], action_log_probs[0], action_to_take[0]
    
    # def evaluate_actions(self, state_tensor, actions_tensor):
    #     value, mu, cov = self.value_action_predictor(state_tensor)
    #     dist = MultivariateNormal(mu, cov)
    #     action_log_probs = dist.log_prob(actions_tensor)
    #
    #     return value, action_log_probs

    def transform(self, state):
        """
        Take the JointState to tensors
        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device)
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states]). \
            to(self.device)

        return robot_state_tensor, human_states_tensor

