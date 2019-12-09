import logging
import abc
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta


class MPRLTrainer(object):
    def __init__(self, value_estimator, state_predictor, memory, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor, share_graph_model):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device
        self.writer = writer
        self.target_policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor
        self.share_graph_model = share_graph_model
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

        if self.state_predictor.trainable:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters()) +
                 list(self.state_predictor.named_parameters())]), self.optimizer_str))
        else:
            logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
                [name for name, param in list(self.value_estimator.named_parameters())]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))

            update_counter = 0
            for data in self.data_loader:
                robot_states, human_states, values, _, _, next_human_states = data

                # optimize value estimator
                self.v_optimizer.zero_grad()
                outputs = self.value_estimator((robot_states, human_states))
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                epoch_v_loss += loss.data.item()

                # optimize state predictor
                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        _, next_human_states_est = self.state_predictor((robot_states, human_states), None)
                        loss = self.criterion(next_human_states_est, next_human_states)
                        loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss += loss.data.item()
                    update_counter += 1

            logging.debug('{}-th epoch ends'.format(epoch))
            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)
            logging.info('Average loss in epoch %d: %.2E, %.2E', epoch, epoch_v_loss / len(self.memory),
                         epoch_s_loss / len(self.memory))

        return

    def optimize_batch(self, num_batches, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        s_losses = 0
        batch_count = 0
        for data in self.data_loader:
            robot_states, human_states, _, rewards, next_robot_states, next_human_states = data

            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator((robot_states, human_states))

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model((next_robot_states, next_human_states))

            # values = values.to(self.device)
            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.v_optimizer.step()
            v_losses += loss.data.item()

            # optimize state predictor
            if self.state_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), None,
                                                                    detach=self.detach_state_predictor)
                    loss = self.criterion(next_human_states_est, next_human_states)
                    loss.backward()
                    self.s_optimizer.step()
                    s_losses += loss.data.item()

            batch_count += 1
            if batch_count > num_batches:
                break

        average_v_loss = v_losses / num_batches
        average_s_loss = s_losses / num_batches
        logging.info('Average loss : %.2E, %.2E', average_v_loss, average_s_loss)
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)

        return average_v_loss, average_s_loss


class VNRLTrainer(object):
    def __init__(self, model, memory, device, policy, batch_size, optimizer_str, writer):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.optimizer = None
        self.writer = writer

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in self.model.named_parameters()]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                inputs, values, _, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            average_epoch_loss = epoch_loss / len(self.memory)
            self.writer.add_scalar('IL/average_epoch_loss', average_epoch_loss, epoch)
            logging.info('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, collate_fn=pad_batch)
        losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, _, rewards, next_states = data
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model(next_states)

            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_loss = losses / num_batches
        logging.info('Average loss : %.2E', average_loss)

        return average_loss
    
    
class RglACTrainer(object):
    def __init__(self, model, memory, device, policy, batch_size, optimizer_str, writer):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.policy = policy
        self.target_model = None
        self.criterion_val = nn.MSELoss().to(device)
        self.criterion_pi = nn.CrossEntropyLoss().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.optimizer = None
        self.writer = writer

        # for value update
        self.gamma = 0.9
        self.time_step = 0.25
        self.v_pref = 1
        self.clip_param = 0.1
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise NotImplementedError
        logging.info('Lr: {} for parameters {} with {} optimizer'.format(learning_rate, ' '.join(
            [name for name, param in self.model.named_parameters()]), self.optimizer_str))

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            # convert action into indices            
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_value_loss = 0
        average_policy_loss = 0
        for epoch in range(num_epochs):
            value_loss = 0
            policy_loss = 0
            logging.debug('{}-th epoch starts'.format(epoch))
            for data in self.data_loader:
                inputs, values, _, actions = data
                self.optimizer.zero_grad()
                # outputs_val, outputs_mu, outputs_cov = self.model(inputs)
                # action_log_probs = MultivariateNormal(outputs_mu, outputs_cov).log_prob(actions)
                outputs_val, alpha_beta_1, alpha_beta_2 = self.model(inputs)
                vx_dist = Beta(alpha_beta_1[:, 0], alpha_beta_1[:, 1])
                vy_dist = Beta(alpha_beta_2[:, 0], alpha_beta_2[:, 1])
                actions = torch.cat([vx_dist.sample().unsqueeze(1), vy_dist.sample().unsqueeze(1)], dim=1)
                action_log_probs = vx_dist.log_prob(actions[:, :1]) + vy_dist.log_prob(actions[:, 1:])

                values = values.to(self.device)

                loss1 = self.criterion_val(outputs_val, values)
                loss2 = -action_log_probs.mean()
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()
                value_loss += loss1.data.item()
                policy_loss += loss2.data.item()
            logging.debug('{}-th epoch ends'.format(epoch))
            average_value_loss = value_loss / len(self.memory)
            average_policy_loss = policy_loss / len(self.memory)
            self.writer.add_scalar('IL/average_value_loss', average_value_loss, epoch)
            self.writer.add_scalar('IL/average_policy_loss', average_policy_loss, epoch)
            logging.info('Average value, policy loss in epoch %d: %.2E, %.2E', epoch,
                         average_value_loss, average_policy_loss)

        return average_value_loss

    def optimize_batch(self, num_batches, episode=None):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        value_losses = 0
        policy_losses = 0
        entropy = 0
        l2_losses = 0
        batch_count = 0
        for data in self.data_loader:
            inputs, values, rewards, actions, returns, old_action_log_probs, adv_targ = data
            self.optimizer.zero_grad()
            # outputs_vals, outputs_mu, outputs_cov = self.model(inputs)
            # dist = MultivariateNormal(outputs_mu, outputs_cov)
            # action_log_probs = dist.log_prob(actions)
            outputs_vals, alpha_beta_1, alpha_beta_2 = self.model(inputs)
            vx_dist = Beta(alpha_beta_1[:, 0], alpha_beta_1[:, 1])
            vy_dist = Beta(alpha_beta_2[:, 0], alpha_beta_2[:, 1])
            actions = torch.cat([vx_dist.sample().unsqueeze(1), vy_dist.sample().unsqueeze(1)], dim=1)
            action_log_probs = vx_dist.log_prob(actions[:, :1]) + vy_dist.log_prob(actions[:, 1:])

            # TODO: check why entropy is negative
            dist_entropy = vx_dist.entropy().mean() + vy_dist.entropy().mean()
            
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
            loss1 = -torch.min(surr1, surr2).mean()
            loss2 = self.criterion_val(outputs_vals, values) * 0.5 * self.value_loss_coef
            loss3 = - dist_entropy * self.entropy_coef

            # speed_square_diff = torch.sum(torch.pow(outputs_mu, 2), dim=1) - torch.Tensor([1]).to(self.device).double()
            # loss4 = torch.pow(torch.max(speed_square_diff, torch.Tensor([0]).to(self.device).double()), 2).mean() * 1

            loss = loss1 + loss2 + loss3
            loss.backward()
            self.optimizer.step()

            value_losses += loss2.data.item()
            policy_losses += loss1.data.item()
            entropy += float(dist_entropy.cpu())
            # l2_losses += loss4.data.item()
            batch_count += 1
            if batch_count > num_batches:
                break

        average_value_loss = value_losses / num_batches
        average_policy_loss = policy_losses / num_batches
        average_entropy = entropy / num_batches
        average_l2_loss = l2_losses / num_batches
        logging.info('Average value, policy loss : %.2E, %.2E', average_value_loss, average_policy_loss)
        self.writer.add_scalar('train/average_value_loss', average_value_loss, episode)
        self.writer.add_scalar('train/average_policy_loss', average_policy_loss, episode)
        self.writer.add_scalar('train/average_entropy', average_entropy, episode)
        # self.writer.add_scalar('train/average_l2_loss', average_l2_loss, episode)

        return average_value_loss


def pad_batch(batch):
    """
    args:
        batch - list of (tensor, label)
    return:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    def sort_states(position):
        # sort the sequences in the decreasing order of length
        sequences = sorted([x[position] for x in batch], reverse=True, key=lambda t: t.size()[0])
        packed_sequences = torch.nn.utils.rnn.pack_sequence(sequences)
        return torch.nn.utils.rnn.pad_packed_sequence(packed_sequences, batch_first=True)

    states = sort_states(0)
    values = torch.cat([x[1] for x in batch]).unsqueeze(1)
    rewards = torch.cat([x[2] for x in batch]).unsqueeze(1)
    next_states = sort_states(3)

    return states, values, rewards, next_states

