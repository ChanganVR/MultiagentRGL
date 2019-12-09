import os
import logging
import copy

import torch
import numpy as np
from tqdm import tqdm

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionXY


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
        stats = {'success': 0,
                 'collision': 0,
                 'timeout': 0,
                 'discomfort': 0}
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
            episode_done = False
            agent_states = [[] for _ in range(len(self.agents))]
            agent_values = [[] for _ in range(len(self.agents))]
            agent_actions = [[] for _ in range(len(self.agents))]
            agent_rewards = [[] for _ in range(len(self.agents))]
            agent_action_log_probs = [[] for _ in range(len(self.agents))]
            past_successes = np.zeros(len(self.agents)).astype(np.bool)
            past_success_time = np.zeros(len(self.agents)) + self.env.time_limit
            while not episode_done:
                # TODO: ORCA as policy in test
                if imitation_learning:
                    with torch.no_grad():
                        actions_to_take = [agent.act(ob) for agent, ob in zip(self.agents, obs)]
                    for k in range(len(self.agents)):
                        agent_actions[k].append(actions_to_take[k])
                else: #for ppo
                    actions_to_take = []
                    for k, (agent, ob) in enumerate(zip(self.agents, obs)):
                        if not past_successes[k]:
                            with torch.no_grad():
                                value, action, action_log_probs, _action_to_take = agent.act(ob)
                            agent_values[k].append(value[0].item())
                            agent_actions[k].append(action)
                            agent_action_log_probs[k].append(action_log_probs)
                            actions_to_take.append(_action_to_take)
                        else:
                            actions_to_take.append(ActionXY(0, 0))

                obs, rewards, dones, infos = self.env.step(actions_to_take)
                for k in range(len(self.agents)):
                    if not past_successes[k]:
                        agent_states[k].append(self.agents[k].last_state)
                        agent_rewards[k].append(rewards[k])

                # episodes terminate
                timeout = any([isinstance(info, Timeout) for info in infos])
                all_success = all([isinstance(info, ReachGoal) for info in infos])
                any_collision = any([isinstance(info, Collision) for info in infos])
                if timeout or all_success or any_collision:
                    episode_done = True

                current_successes = [isinstance(info, ReachGoal) for info in infos]
                for k in range(len(self.agents)):
                    if current_successes[k] and not past_successes[k]:
                        past_success_time[k] = self.env.global_time
                past_successes = np.bitwise_or(past_successes, np.array(current_successes))

            stats['success'] += np.average(past_successes)
            success_times.append(np.average([past_success_time[k] if past_successes[k] else 0
                                             for k in range(len(self.agents))]))

            stats['collision'] += np.average([isinstance(info, Collision) for info in infos])
            if any_collision:
                collision_cases.append(i)

            stats['timeout'] += np.average([isinstance(info, Timeout) for info in infos])
            timeout_cases.append(i)
            timeout_times.append(self.env.time_limit)
            # if isinstance(info, ReachGoal):
            #     success += 1
            #     success_times.append(self.env.global_time)
            # elif isinstance(info, Collision):
            #     collision += 1
            #     collision_cases.append(i)
            #     collision_times.append(self.env.global_time)
            # elif isinstance(info, Timeout):
            #     timeout += 1
            #     timeout_cases.append(i)
            #     timeout_times.append(self.env.time_limit)
            # else:
            #     raise ValueError('Invalid end signal from environment')

            if update_memory:
                # TODO: try different strategies
                for k in range(len(self.agents)):
                    # skip experience that doesn't lead to collision
                    if any_collision and isinstance(infos[k], Nothing):
                        continue
                    if imitation_learning:
                        self.update_memory(agent_states[k], agent_rewards[k], agent_actions[k],
                                           imitation_learning=imitation_learning)
                    else:
                        self.update_memory(agent_states[k], agent_rewards[k], agent_actions[k],
                                           agent_values[k], agent_action_log_probs[k])

            agent_cumulative_rewards = list()
            agent_average_returns = list()
            for k in range(len(self.agents)):
                agent_cumulative_rewards.append(sum([pow(self.gamma, t * self.time_step * self.v_pref)
                                                * reward for t, reward in enumerate(agent_rewards[k])]))
                returns = []
                for step in range(len(agent_rewards)):
                    step_return = sum([pow(self.gamma, t * self.time_step * self.v_pref)
                                       * reward for t, reward in enumerate(agent_rewards[k][step:])])
                    returns.append(step_return)
                agent_average_returns.append(average(returns))
            cumulative_rewards.append(np.average(agent_cumulative_rewards))
            average_returns.append(np.average(agent_average_returns))

            if pbar:
                pbar.update(1)
        success_rate = stats['success'] / num_episode
        collision_rate = stats['collision'] / num_episode
        timeout_rate = stats['timeout'] / num_episode
        # assert stats['success'] + stats['collision'] + stats['timeout'] == num_episode
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
                         stats['discomfort'] / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), \
                          average(average_returns), timeout_rate

        return self.statistics

    def update_memory(self, states, rewards, actions, values=None, action_log_probs=None, imitation_learning=False):
        assert len(states) == len(rewards)
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        _gamma = pow(self.gamma, self.time_step * self.v_pref)
        gae = 0
        returns_list = []

        if imitation_learning:
            actions = [((action[0] + 1) / 2, (action[1] + 1) / 2) for action in actions]

        tmp_tuples = list()
        for i, state in reversed(list(enumerate(states))):
            # VALUE UPDATE
            if imitation_learning:

                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.time_step * self.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])

                action = torch.from_numpy(np.array(actions[i])).to(self.device)
                value = torch.Tensor([value]).to(self.device)
                reward = torch.Tensor([rewards[i]]).to(self.device)
                tmp_tuples.append((state, value, reward, action))
            else:
                if i == len(states) - 1:
                    td_target = rewards[i]
                else:
                    td_target = rewards[i] + _gamma * values[i + 1]
                delta = td_target - values[i]
                gae = delta + _gamma * gae  # * gae_lambda
                reward_to_go = gae + values[i] # i.e., return
                returns_list.append(reward_to_go)
                
                reward_to_go = torch.Tensor([reward_to_go]).to(self.device)
                value = torch.Tensor([td_target]).to(self.device)
                reward = torch.Tensor([rewards[i]]).to(self.device)
                tmp_tuples.append((state, value, reward, actions[i], reward_to_go, action_log_probs[i]))
        
        returns_list = list(reversed(returns_list))
        advantages = torch.FloatTensor([[returns_list[i] - values[i]] for i in range(len(returns_list))]).to(self.device)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for i, t in enumerate(tmp_tuples):
            if imitation_learning:
                self.memory.push(t)
            else:
                self.memory.push(t + (advantages[i],))

        # for i in range(len(returns_list)):
        #     self.memory.memory[i] = self.memory.memory[i] + (advantages[i],)

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return, timeout = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)
        self.writer.add_scalar(tag_prefix + '/timeout', timeout, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
