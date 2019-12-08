import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.utils.multiagent_explorer import MultiagentExplorer
from crowd_sim.envs.multiagent_sim import MultiagentSim


def main(args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()

    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env = MultiagentSim()
    env.configure(env_config)
    agents = env.agents
    for agent in agents:
        agent.set_policy(policy)
    if args.square:
        env.test_scenario = 'square_crossing'
    if args.circle:
        env.test_scenario = 'circle_crossing'
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario

    explorer = MultiagentExplorer(env, device, None, gamma=0.9)
    policy.set_phase(args.phase)
    policy.set_device(device)

    if args.visualize:
        explorer.run_episodes(1, args.phase)
        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_dir is not None:
                args.video_file = os.path.join(args.video_dir, policy_config.name)
                args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        logging.info('It takes %.2f seconds to finish.', env.global_time)
    else:
        explorer.run_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            if args.model_dir is not None:
                plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            else:
                plt.show()
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='rgl')
    parser.add_argument('-m', '--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
