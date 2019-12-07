import os
import logging
import copy
import torch
from tqdm import tqdm
from crowd_sim.envs.utils.info import *

########
class MultiagentExplorer(object):
    def __init__(self, env, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.agents = env.agents
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.statistics = None

        self.time_step = self.agents[0].time_step
        self.v_pref = self.agents[0].v_pref

    # @profile
    def run_episodes(self, num_episode, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                     print_failure=False):
        for agent in self.agents:
            agent.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        collision_cases = []
        timeout_cases = []

        if num_episode != 1:
            pbar = tqdm(total=num_episode)
        else:
            pbar = None

        for i in range(num_episode):
            obs = self.env.reset(phase)
            done = False
            agent_states = [[] for _ in range(len(self.agents))]
            agent_values = [[] for _ in range(len(self.agents))]
            agent_actions = [[] for _ in range(len(self.agents))]
            agent_action_log_probs = [[] for _ in range(len(self.agents))]
            rewards = []
            while not done:
                if imitation_learning or phase in ['val', 'test']:
                    with torch.no_grad():
                        action_to_take = [agent.act(ob) for agent, ob in zip(self.agents, obs)]
                else: #for ppo
                    for k, (agent, ob) in enumerate(zip(self.agents, obs)):
                        with torch.no_grad():
                            value, action, action_log_probs, action_to_take = agent.act(ob)
                        agent_values[k].append(value)
                        agent_actions[k].append(action)
                        agent_action_log_probs[k].append(action_log_probs)
                        
                obs, reward, done, info = self.env.step(action_to_take)
                for k in range(len(self.agents)):
                    agent_states[k].append(self.agents[k].last_state)

                rewards.append(reward)

                if isinstance(info, Discomfort):
                    discomfort += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    for k in range(len(self.agents)):
                        self.update_memory(agent_states[k], rewards, imitation_learning=imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.time_step * self.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.time_step * self.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)
        success_rate = success / num_episode
        collision_rate = collision / num_episode
        assert success + collision + timeout == num_episode
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, average(cumulative_rewards),
                                                       average(average_returns)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times)
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         discomfort / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(average_returns)

        return self.statistics

    def update_memory(self, states, rewards, actions, values=None, action_log_probs=None, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.time_step * self.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
            value = torch.Tensor([value]).to(self.device)
            reward = torch.Tensor([rewards[i]]).to(self.device)

            self.memory.push((state, value, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
