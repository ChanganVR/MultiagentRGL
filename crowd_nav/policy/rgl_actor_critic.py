import logging

import torch
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.graph_model import RGL
from crowd_nav.policy.value_action_predictor import ValueActionPredictor


class RglActorCritic(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'RglActorCritic'
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
        self.share_graph_model = None
        self.value_action_predictor = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.do_action_clip = config.rgl_actor_critic.do_action_clip        
        self.share_graph_model = config.rgl_actor_critic.share_graph_model

        if self.share_graph_model:
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_action_predictor = ValueActionPredictor(config, graph_model)            
            self.model = [graph_model, self.value_action_predictor.value_network, \
                          self.value_action_predictor.action_network]
        else:
            graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
            graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_action_predictor = ValueActionPredictor(config, graph_model1, graph_model2, False)            
            self.model = [graph_model1, graph_model2, self.value_action_predictor.value_network, \
                          self.value_action_predictor.action_network]

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
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_action_predictor

    def get_state_dict(self):
        if self.share_graph_model:
            return {
                'graph_model': self.value_action_predictor.graph_model.state_dict(),
                'value_network': self.value_action_predictor.value_network.state_dict(),
                'action_network': self.value_action_predictor.action_network.state_dict()
            }
        else:
            return {
                'graph_model1': self.value_action_predictor.graph_model_val.state_dict(),
                'graph_model2': self.value_action_predictor.graph_model_act.state_dict(),
                'value_network': self.value_action_predictor.value_network.state_dict(),
                'action_network': self.value_action_predictor.action_network.state_dict()
            }


    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):        
        if self.share_graph_model:
            self.value_action_predictor.graph_model.load_state_dict(state_dict['graph_model'])
        else:
            self.value_action_predictor.graph_model_val.load_state_dict(state_dict['graph_model1'])
            self.value_action_predictor.graph_model_act.load_state_dict(state_dict['graph_model2'])

        self.value_action_predictor.value_network.load_state_dict(state_dict['value_network'])
        self.value_action_predictor.action_network.load_state_dict(state_dict['action_network'])


    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for j, speed in enumerate(speeds):
            if j == 0:
                # index for action (0, 0)
                self.action_group_index.append(0)
            # only two groups in speeds
            if j < 3:
                speed_index = 0
            else:
                speed_index = 1

            for i, rotation in enumerate(rotations):
                rotation_index = i // 2

                action_index = speed_index * self.sparse_rotation_samples + rotation_index
                self.action_group_index.append(action_index)

                if holonomic:
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
                else:
                    action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def act(self, state_tensor): #state is a batch of tensors rather than a joint state
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)
            
        value, action_feat = self.value_action_predictor(state_tensor)
        dist = FixedCategorical(action_feat)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        action_to_take = self.action_index_to_action(action)
        
        return value, action, action_log_probs, action_to_take
    
    def predict(self, state): #used in Agent class; state must be a joint state
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
        _, _, _, action_to_take = self.act(state_tensor)
        if self.phase == 'train':
            self.last_state = self.transform(state)
            
        return action_to_take[0]
    
    def action_index_to_action(self, indices): #indices: (batch_size, 1) tensor
        action_to_take = []
        batch_size = list(indices.size())[0]
        for i in range(batch_size):
            action_to_take.append(self.action_space[indices[i,0].item()])
        return action_to_take 
    

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
    
    
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
