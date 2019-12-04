import logging
import random
import math

import gym
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist


class MultiagentSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.group_num = None
        self.group_size = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.save_scene_dir = None
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []

        self.phase = None

        self.num_agent = None
        self.agents = None

    def configure(self, config, robot):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty

        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        self.human_num = config.sim.human_num
        self.agents = [robot] + [Robot(config, 'robot') for _ in range(self.human_num)]
        for agent in self.agents[1:]:
            agent.policy = robot.policy
            agent.kinematics = robot.kinematics

    def reset_agent_states(self):
        self.agents[0].set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        self.agents[1].set(-self.circle_radius, 0, self.circle_radius, 0, 0, 0, 0)
        self.agents[2].set(self.circle_radius, 0, -self.circle_radius, 0, 0, 0, 0)
        self.agents[3].set(0, self.circle_radius, 0, -self.circle_radius, 0, 0, np.pi / 2)
        self.agents[4].set(self.circle_radius * 0.71, self.circle_radius * 0.71,
                           -self.circle_radius * 0.71, -self.circle_radius * 0.71, 0, 0, 0)
        self.agents[5].set(-self.circle_radius * 0.71, -self.circle_radius * 0.71,
                           self.circle_radius * 0.71, self.circle_radius * 0.71, 0, 0, 0)
        # if self.current_scenario == 'circle_crossing':
        #     while True:
        #         angle = np.random.random() * np.pi * 2
        #         # add some noise to simulate all the possible cases robot could meet with human
        #         px_noise = (np.random.random() - 0.5) * human.v_pref
        #         py_noise = (np.random.random() - 0.5) * human.v_pref
        #         px = self.circle_radius * np.cos(angle) + px_noise
        #         py = self.circle_radius * np.sin(angle) + py_noise
        #         collide = False
        #         for agent in [self.robot] + self.humans:
        #             min_dist = human.radius + agent.radius + self.discomfort_dist
        #             if norm((px - agent.px, py - agent.py)) < min_dist or \
        #                     norm((px - agent.gx, py - agent.gy)) < min_dist:
        #                 collide = True
        #                 break
        #         if not collide:
        #             break
        #     human.set(px, py, -px, -py, 0, 0, 0)

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0

        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                     'val': 0, 'test': self.case_capacity['val']}

        if self.case_counter[phase] >= 0:
            np.random.seed(base_seed[phase] + self.case_counter[phase])
            random.seed(base_seed[phase] + self.case_counter[phase])
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            self.reset_agent_states()
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        for agent in self.agents:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        # get current observation
        obs = self.compute_observations()

        return obs

    def step(self, actions):
        # collision detection between all agents
        num_agent = len(self.agents)
        collision = False
        for i in range(num_agent):
            for j in range(i + 1, num_agent):
                dx = self.agents[i].px - self.agents[j].px
                dy = self.agents[i].py - self.agents[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.agents[i].radius - self.agents[j].radius
                if dist < 0:
                    collision = True

        # check if all agents reach goals
        reaching_goals = list()
        for i, action in enumerate(actions):
            agent = self.agents[i]
            end_position = np.array(agent.compute_position(action, self.time_step))
            reaching_goal = norm(end_position - np.array(agent.get_goal_position())) < agent.radius
            reaching_goals.append(reaching_goal)

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif all(reaching_goals):
            reward = self.success_reward
            done = True
            info = ReachGoal()
        else:
            reward = 0
            done = False
            info = Nothing()

        # update all agents
        for i, action in enumerate(actions):
            self.agents[i].step(action)

        self.global_time += self.time_step
        obs = self.compute_observations()

        return obs, reward, done, info

    def compute_observations(self):
        obs = list()
        for agent in self.agents:
            ob = list()
            for other_agent in self.agents:
                if other_agent != agent:
                    ob.append(other_agent.get_observable_state())
            obs.append(ob)

        return obs

    def render(self, output_file=None):
        pass
