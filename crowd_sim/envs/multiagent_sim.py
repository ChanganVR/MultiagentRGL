import logging
import random
import math
from itertools import product

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

    def configure(self, config):
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
        self.agents = [Robot(config, 'robot') for _ in range(self.human_num)]
        for agent in self.agents:
            agent.time_step = self.time_step
        # for agent in self.agents[1:]:
        #     agent.policy = robot.policy
        #     agent.kinematics = robot.kinematics

    def reset_agent_states(self):
        # self.agents[0].set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        # self.agents[1].set(-self.circle_radius, 0, self.circle_radius, 0, 0, 0, 0)
        # self.agents[2].set(self.circle_radius, 0, -self.circle_radius, 0, 0, 0, 0)
        # self.agents[3].set(0, self.circle_radius, 0, -self.circle_radius, 0, 0, np.pi / 2)
        # self.agents[4].set(self.circle_radius * 0.71, self.circle_radius * 0.71,
        #                    -self.circle_radius * 0.71, -self.circle_radius * 0.71, 0, 0, 0)
        # self.agents[5].set(-self.circle_radius * 0.71, -self.circle_radius * 0.71,
        #                    self.circle_radius * 0.71, self.circle_radius * 0.71, 0, 0, 0)

        if self.current_scenario == 'circle_crossing':
            existing_agents = list()
            for agent in self.agents:
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    px_noise = (np.random.random() - 0.5) * agent.v_pref
                    py_noise = (np.random.random() - 0.5) * agent.v_pref
                    px = self.circle_radius * np.cos(angle) + px_noise
                    py = self.circle_radius * np.sin(angle) + py_noise
                    collide = False
                    for existing_agent in existing_agents:
                        min_dist = agent.radius + existing_agent.radius
                        if norm((px - existing_agent.px, py - existing_agent.py)) < min_dist or \
                                norm((px - existing_agent.gx, py - existing_agent.gy)) < min_dist:
                            collide = True
                            break
                    if not collide:
                        break
                agent.set(px, py, -px, -py, 0, 0, 0)
                existing_agents.append(agent)

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
            self.current_scenario = 'circle_crossing'
            self.reset_agent_states()
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            raise NotImplementedError

        for agent in self.agents:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()

        # get current observation
        obs = self.compute_observations()

        return obs

    def step(self, actions):
        # collision detection between all agents
        num_agent = len(self.agents)
        collisions = [False] * num_agent
        for i, j in product(range(num_agent), range(num_agent)):
            if i == j:
                continue
            dx = self.agents[i].px - self.agents[j].px
            dy = self.agents[i].py - self.agents[j].py
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.agents[i].radius - self.agents[j].radius
            if dist < 0:
                collisions[i] = True
                collisions[j] = True

        # check if all agents reach goals
        reaching_goals = list()
        for i, action in enumerate(actions):
            agent = self.agents[i]
            end_position = np.array(agent.compute_position(action, self.time_step))
            reaching_goal = norm(end_position - np.array(agent.get_goal_position())) < agent.radius
            reaching_goals.append(reaching_goal)

        rewards = np.zeros(num_agent)
        dones = np.zeros(num_agent)
        infos = [None] * num_agent
        for i in range(num_agent):
            if self.global_time >= self.time_limit - 1:
                rewards[i] = 0
                dones[i] = True
                infos[i] = Timeout()
            elif collisions[i]:
                rewards[i] = self.collision_penalty
                dones[i] = True
                infos[i] = Collision()
            elif reaching_goals[i]:
                rewards[i] = self.success_reward
                dones[i] = True
                infos[i] = ReachGoal()
            else:
                rewards[i] = 0
                dones[i] = False
                infos[i] = Nothing()

        # update all agents
        for i, action in enumerate(actions):
            self.agents[i].step(action)
        self.states.append([agent.get_full_state() for agent in self.agents])

        self.global_time += self.time_step
        obs = self.compute_observations()

        return obs, rewards, dones, infos

    def compute_observations(self):
        obs = list()
        for agent in self.agents:
            ob = list()
            for other_agent in self.agents:
                if other_agent != agent:
                    ob.append(other_agent.get_observable_state())
            obs.append(ob)

        return obs

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            plot_scale = 1.2
            ax.set_xlim(-self.panel_width * plot_scale / (2 * self.panel_scale),
                        self.panel_width * plot_scale / (2 * self.panel_scale))
            ax.set_ylim(-self.panel_height * plot_scale / (2 * self.panel_scale),
                        self.panel_height * plot_scale / (2 * self.panel_scale))
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            plot_human_radius = 0.3
            plot_robot_radius = 0.3
            human_positions = [[state[1][j].position for j in range(len(state[1]))] for state in self.states]
            human_ids = [[state[2][j] for j in range(len(state[2]))] for state in self.states]

            human_colors = [cmap(i) for i in range(len(self.human_starts))]

            for h in range(len(self.human_starts)):
                human_start = mlines.Line2D([self.human_starts[h][0]], [self.human_starts[h][1]], color='b', marker='o',
                                            linestyle='None', markersize=4)
                human_goal = mlines.Line2D([self.human_goals[h][0]], [self.human_goals[h][1]], color='r', marker='*',
                                           linestyle='None', markersize=4)

                ax.add_artist(human_start)
                ax.add_artist(human_goal)

            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], plot_robot_radius, fill=False, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            # plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=14)

            # add humans and their numbers
            # Sha: here humans refer to the humans in the first frame
            humans = [plt.Circle(human_positions[0][i], plot_human_radius, fill=False, color='r')
                      for i in range(self.dynamic_human_num[0])]
            # disable showing human numbers
            if display_numbers:
                human_numbers = [
                    plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(human_ids[0][i]),
                             color='black') for i in range(self.dynamic_human_num[0])]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(0.1, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)

            # add human counter
            count_human = plt.text(0.6, 0.9, 'Count Human: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(count_human)

            # add reward displayer
            reward_displayer = plt.text(0.9, 1, 'r(s,a) is {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(reward_displayer)

            # add robot velocity displayer
            robot_velocity = plt.text(0.1, 1, 'v:{}, vx:{}, vy:{}'.format(0, 0, 0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(robot_velocity)

            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = plot_robot_radius
            orientations = []
            # sha: deal with the dynamic human number issue:
            time_step = 0
            for state in self.states:
                orientation = []
                for i in range(self.dynamic_human_num[time_step] + 1):
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                    if time_step == 0:
                        if i == 0:
                            arrow_color = 'black'
                            arrows = [
                                patches.FancyArrowPatch(*orientation[i], color=arrow_color, arrowstyle=arrow_style)]
                        else:
                            arrows.extend([patches.FancyArrowPatch(*orientation[i])])
                time_step += 1
                orientations.append(orientation)

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False

            # add agent start positions and goals
            agent_colors = [cmap(i) for i in range(len(self.agents))]
            for i in range(len(self.agents)):
                agent = self.agents[i]
                agent_goal = mlines.Line2D([agent.get_goal_position()[0]], [agent.get_goal_position()[1]],
                                           color=agent_colors[i],
                                           marker='*', linestyle='None', markersize=8)
                ax.add_artist(agent_goal)
                agent_start = mlines.Line2D([agent.get_start_position()[0]], [agent.get_start_position()[1]],
                                            color=agent_colors[i],
                                            marker='o', linestyle='None', markersize=8)
                ax.add_artist(agent_start)

            # add agents and their numbers
            agent_positions = [[state[j].position for j in range(len(self.agents))] for state in self.states]
            agents = [plt.Circle(agent_positions[0][i], self.agents[i].radius, fill=False, color=cmap(i))
                      for i in range(len(self.agents))]

            # disable showing human numbers
            if display_numbers:
                agent_numbers = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.agents))]

            for i, agent in enumerate(agents):
                ax.add_artist(agent)
                if display_numbers:
                    ax.add_artist(agent_numbers[i])

            # add time annotation
            time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)

            global_step = 0

            def update(frame_num):
                nonlocal global_step
                # nonlocal arrows
                global_step = frame_num

                for i, agent in enumerate(agents):
                    agent.center = agent_positions[frame_num][i]
                    if display_numbers:
                        agent_numbers[i].set_position((agent.center[0] - x_offset, agent.center[1] + y_offset))
                # for arrow in arrows:
                #     arrow.remove()

                # for i in range(self.human_num + 1):
                #     orientation = orientations[i]
                #     if i == 0:
                #         arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                #                                           arrowstyle=arrow_style)]
                #     else:
                #         arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                #                                                arrowstyle=arrow_style)])
                #
                # for arrow in arrows:
                #     ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_sco res[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        else:
            raise NotImplementedError
