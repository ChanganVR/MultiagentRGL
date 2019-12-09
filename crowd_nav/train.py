import sys
import logging
import argparse
import os
import shutil
import importlib.util
import torch
import gym
import copy
import git
import re
from tensorboardX import SummaryWriter
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import VNRLTrainer, MPRLTrainer, RglACTrainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.multiagent_explorer import MultiagentExplorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.multiagent_sim import MultiagentSim


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    set_random_seeds(args.randomseed)
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            key = input('Output directory already exists! Overwrite the folder? (y/n)')
            if key == 'y' and not args.resume:
                shutil.rmtree(args.output_dir)
            else:
                make_new_dir = False
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))

        # # insert the arguments from command line to the config file
        # with open(os.path.join(args.output_dir, 'config.py'), 'r') as fo:
        #     config_text = fo.read()
        # search_pairs = {r"gcn.X_dim = \d*": "gcn.X_dim = {}".format(args.X_dim),
        #                 r"gcn.num_layer = \d": "gcn.num_layer = {}".format(args.layers),
        #                 r"gcn.similarity_function = '\w*'": "gcn.similarity_function = '{}'".format(args.sim_func),
        #                 r"gcn.layerwise_graph = \w*": "gcn.layerwise_graph = {}".format(args.layerwise_graph),
        #                 r"gcn.skip_connection = \w*": "gcn.skip_connection = {}".format(args.skip_connection)}
        #
        # for find, replace in search_pairs.items():
        #     config_text = re.sub(find, replace, config_text)
        #
        # with open(os.path.join(args.output_dir, 'config.py'), 'w') as fo:
        #     fo.write(config_text)

    args.config = os.path.join(args.output_dir, 'config.py')
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))
    logging.info('Current config content is :{}'.format(config))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    writer = SummaryWriter(log_dir=args.output_dir)

    # configure policy
    policy_config = config.PolicyConfig()
    policy = policy_factory[policy_config.name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env = MultiagentSim()
    env.configure(env_config)
    agents = env.agents

    # read training parameters
    train_config = config.TrainConfig(args.debug)
    rl_learning_rate = train_config.train.rl_learning_rate
    train_batches = train_config.train.train_batches
    train_episodes = train_config.train.train_episodes
    sample_episodes = train_config.train.sample_episodes
    target_update_interval = train_config.train.target_update_interval
    evaluation_interval = train_config.train.evaluation_interval
    capacity = train_config.train.capacity
    epsilon_start = train_config.train.epsilon_start
    epsilon_end = train_config.train.epsilon_end
    epsilon_decay = train_config.train.epsilon_decay
    checkpoint_interval = train_config.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.trainer.batch_size
    optimizer = train_config.trainer.optimizer
    trainer = RglACTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    explorer = MultiagentExplorer(env, device, writer, memory, policy.gamma, target_policy=policy)

    if os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        il_episodes = train_config.imitation_learning.il_episodes
        il_policy = train_config.imitation_learning.il_policy
        il_epochs = train_config.imitation_learning.il_epochs
        il_learning_rate = train_config.imitation_learning.il_learning_rate
        trainer.set_learning_rate(il_learning_rate)

        il_policy = policy_factory[il_policy]()
        il_policy.multiagent_training = policy.multiagent_training
        il_policy.safety_space = train_config.imitation_learning.safety_space
        for agent in agents:
            agent.set_policy(il_policy)
        explorer.run_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        policy.save_model(il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    trainer.update_target_model(model)

    # reinforcement learning
    # TODO: add it to config
    # update memory
    memory = ReplayMemory(200)
    trainer = RglACTrainer(model, memory, device, policy, batch_size, optimizer, writer)
    trainer.set_learning_rate(rl_learning_rate)
    explorer.memory = memory
    for agent in agents:
        agent.set_policy(policy)

    episode = 0
    best_val_reward = -1
    best_val_model = None

    val_size = env.case_size['val']
    test_size = env.case_size['test']
    # evaluate the model after imitation learning
    if episode % evaluation_interval == 0:
        logging.info('Evaluate the model instantly after imitation learning on the validation cases')
        explorer.run_episodes(val_size, 'val', episode=episode)
        explorer.log('val', episode // evaluation_interval)

        if args.test_after_every_eval:
            explorer.run_episodes(test_size, 'test', episode=episode, print_failure=True)
            explorer.log('test', episode // evaluation_interval)

    episode = 0
    while episode < train_episodes:
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
        policy.set_epsilon(epsilon)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        explorer.log('train', episode)

        trainer.optimize_batch(train_batches, episode)
        episode += 1

        if episode % target_update_interval == 0:
            trainer.update_target_model(model)
        # evaluate the model
        if episode % evaluation_interval == 0:
            _, _, _, reward, _, _ = explorer.run_episodes(val_size, 'val', episode=episode)
            explorer.log('val', episode // evaluation_interval)

            if episode % checkpoint_interval == 0 and reward > best_val_reward:
                best_val_reward = reward
                best_val_model = copy.deepcopy(policy.get_state_dict())
        # test after every evaluation to check how the generalization performance evolves
            if args.test_after_every_eval:
                explorer.run_episodes(test_size, 'test', episode=episode, print_failure=True)
                explorer.log('test', episode // evaluation_interval)

        if episode != 0 and episode % checkpoint_interval == 0:
            current_checkpoint = episode // checkpoint_interval - 1
            save_every_checkpoint_rl_weight_file = rl_weight_file.split('.')[0] + '_' + str(current_checkpoint) + '.pth'
            policy.save_model(save_every_checkpoint_rl_weight_file)

    # # test with the best val model
    if best_val_model is not None:
        policy.load_state_dict(best_val_model)
        torch.save(best_val_model, os.path.join(args.output_dir, 'best_val.pth'))
        logging.info('Save the best val model with the reward: {}'.format(best_val_reward))
    explorer.run_episodes(test_size, 'test', episode=episode, print_failure=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='rgl_ppo')
    parser.add_argument('--config', type=str, default='configs/icra_benchmark/rgl_ppo.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--test_after_every_eval', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=17)

    # arguments for GCN
    # parser.add_argument('--X_dim', type=int, default=32)
    # parser.add_argument('--layers', type=int, default=2)
    # parser.add_argument('--sim_func', type=str, default='embedded_gaussian')
    # parser.add_argument('--layerwise_graph', default=False, action='store_true')
    # parser.add_argument('--skip_connection', default=True, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
