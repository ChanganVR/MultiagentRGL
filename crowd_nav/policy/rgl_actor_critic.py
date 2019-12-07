import numpy as np
import torch

from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.graph_model import RGL
from crowd_nav.policy.value_action_predictor import ValueActionPredictor


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

    def configure(self, config):
        self.set_common_parameters(config)
        graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
        self.value_action_predictor = ValueActionPredictor(config, graph_model)
        self.model = [graph_model, self.value_action_predictor.value_network, \
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

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_action_predictor

    def get_state_dict(self):
        return {
            'graph_model': self.value_action_predictor.graph_model.state_dict(),
            'value_network': self.value_action_predictor.value_network.state_dict(),
            'action_network': self.value_action_predictor.action_network.state_dict()
        }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        self.value_action_predictor.graph_model.load_state_dict(state_dict['graph_model'])
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
            for i, rotation in enumerate(rotations):
                if holonomic:
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
                else:
                    action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def convert_action_to_index(self, memory):
        if self.action_space is None:
            self.build_action_space(v_pref=1)

        for i, mem_tuple in enumerate(memory.memory):
            action = mem_tuple[3]
            min_index = 0
            min_distance = 10000
            for j, a in enumerate(self.action_space):
                distance = np.linalg.norm(np.array(action) - np.array(a))
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            action_index = torch.LongTensor([min_index]).to(self.device)
            new_tuple = mem_tuple[0], mem_tuple[1], mem_tuple[2], action_index, mem_tuple[4]
            memory.memory[i] = new_tuple

    def act(self, state_tensor): # state is a batch of tensors rather than a joint state
        value, action_feat = self.value_action_predictor(state_tensor)
        dist = FixedCategorical(logits=action_feat)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        action_to_take = self.action_index_to_action(action)
        
        return value, action, action_log_probs, action_to_take
    
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
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
        value, action, action_log_probs, action_to_take = self.act(state_tensor)
        if self.phase == 'train':
            self.last_state = self.transform(state)
            
        return value[0], action[0], action_log_probs[0], action_to_take[0]
    
    def evaluate_actions(self, state_tensor, actions_tensor):
        value, action_feat = self.value_action_predictor(state_tensor)
        dist = FixedCategorical(logits=action_feat)
        action_log_probs = dist.log_probs(actions_tensor)
        
        return value, action_log_probs
    
    def action_index_to_action(self, indices): #indices: (batch_size, 1) tensor
        action_to_take = []
        batch_size = list(indices.size())[0]
        for i in range(batch_size):
            action_to_take.append(self.action_space[indices[i, 0].item()])
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