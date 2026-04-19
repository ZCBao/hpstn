import argparse
import gym
import logging
import numpy as np
import os
import random
import shutil
import sys
import torch
import yaml
from tqdm import tqdm
from env.gym_env import GymEnv
from policy.actor_critic import PPOActorCritic
from policy.utils import seed, norm

def test_loop(env, policy, phase='Test', epoch=None, val_test_episodes=100, episode_max_steps=300, gamma=0.99, val_base_seed=500, test_base_seed=0, obs_coord_system='global',
              save_path=None, render=False, save_figure=False, save_gif=False, save_video=False, show_lidar=None, show_trajectory=True):
    if val_test_episodes > 1:
        pbar = tqdm(total=val_test_episodes, desc=phase)
    else:
        pbar = None

    success_time_list, success_speed_list, episode_return_list = [], [], []
    success_episode_list, collision_episode_list, timeout_episode_list = [], [], []
    computation_time_list = []

    for episode in range(val_test_episodes):
        if phase == 'Validation':
            random.seed(val_base_seed + episode)
            np.random.seed(val_base_seed + episode)
        elif phase == 'Test':
            seed(test_base_seed + episode)
            env.timer.start()
        
        obs_list = env.reset()
        last_obs_list = obs_list[:]
        last_last_obs_list = obs_list[:]
        obs_list_list = [[last_last_obs_list[i], last_obs_list[i], obs_list[i]] for i in range(env.robot_num)]

        # render
        if render:
            env.render(save=(save_figure or save_gif or save_video), save_step=0, step_time=env.step_time, show_lidar=show_lidar, show_trajectory=show_trajectory)
            if save_gif or save_video:
                env.env_plot.save_figure(save_path + '/render' + '_r' + str(env.robot_num), test_base_seed + episode, 0, format='png')
        
        robot_speed_list_list = [[] for _ in range(env.robot_num)]
        robot_return_list = [0] * env.robot_num

        for step in range(episode_max_steps):
            raw_act_list = policy.act(obs_list_list)
            act_list = -env.action_space.high + (env.action_space.high - (-env.action_space.high)) * (raw_act_list + 1.0) / 2.0
            if obs_coord_system == 'global':
                vel_list = act_list                    
            else:
                vel_list = []
                for i in range(env.robot_num):
                    robot = env.components["robots"].robot_list[i]
                    if obs_coord_system == 'local_yaw':
                        rot = robot.state[2]
                    elif obs_coord_system == 'local_goal':
                        dgx = robot.goal[0] - robot.state[0]
                        dgy = robot.goal[1] - robot.state[1]
                        rot = np.arctan2(dgy, dgx)
                    vx = act_list[i][0] * np.cos(rot) - act_list[i][1] * np.sin(rot)
                    vy = act_list[i][0] * np.sin(rot) + act_list[i][1] * np.cos(rot)
                    if env.action_space.shape[0] == 2:
                        vel = np.array([vx, vy])
                    elif env.action_space.shape[0] == 3:
                        wz = act_list[i][2]
                        vel = np.array([vx, vy, wz])
                    vel_list.append(vel)

            next_obs_list, reward_list, done_list, info_list = env.env_step(vel_list)

            # update observations and other data
            for i in range(env.robot_num):
                if info_list[i] != -2 and info_list[i] != 2:
                    speed = norm(env.components['robots'].robot_list[i].vel_abs[0:2])
                    robot_speed_list_list[i].append(speed)
                    robot_return_list[i] += pow(gamma, step) * reward_list[i]

            last_last_obs_list = last_obs_list[:]
            last_obs_list = obs_list[:]
            obs_list = next_obs_list[:]
            obs_list_list = [[last_last_obs_list[i], last_obs_list[i], obs_list[i]] for i in range(env.robot_num)]

            # render
            if render:
                env.render(save=(save_figure or save_gif or save_video), save_step=step+1, step_time=env.step_time, show_lidar=show_lidar, show_trajectory=show_trajectory)
                if save_gif or save_video:
                    env.env_plot.save_figure(save_path + '/render' + '_r' + str(env.robot_num), test_base_seed + episode, step + 1, format='png')

            collision = min(info_list) < 0
            success_all = min(info_list) > 0
            tiemout = step == episode_max_steps - 1

            if collision or success_all or tiemout:
                if collision:
                    collision_episode_list.append(episode)
                elif success_all:
                    success_episode_list.append(episode)
                    success_time_list.append((step+1) * env.step_time)
                    success_speed_list.append(np.mean([np.mean(robot_speed_list_list[i]) for i in range(env.robot_num)]))
                elif tiemout:
                    timeout_episode_list.append(episode)
                
                episode_return_list.append(np.mean(robot_return_list))

                if phase == 'Test':
                    time_cost = env.timer.end()
                    computation_time_list.append(time_cost/(step+1))

                # render
                if render:
                    if save_figure:
                        env.env_plot.save_figure(save_path + '/render' + '_r' + str(env.robot_num), test_base_seed + episode, step + 1, format='pdf')
                    if save_gif:
                        env.env_plot.save_gif(save_path + '/render' + '_r' + str(env.robot_num), test_base_seed + episode, env.step_time, clear_figure=True)
                    if save_video:
                        env.env_plot.save_video(save_path + '/render' + '_r' + str(env.robot_num), test_base_seed + episode, env.step_time, clear_figure=True)
                
                if pbar:
                    pbar.update()
                
                break
    
    if pbar:
        pbar.close()
    
    extra_str = '{} after epoch {}'.format(phase, epoch) if epoch else phase
    success_rate = len(success_episode_list)/val_test_episodes
    collision_rate = len(collision_episode_list)/val_test_episodes
    timeout_rate = len(timeout_episode_list)/val_test_episodes
    success_time_avg = np.mean(success_time_list) if success_time_list else 0
    success_time_std = np.std(success_time_list) if success_time_list else 0
    speed_avg = np.mean(success_speed_list) if success_speed_list else 0
    speed_std = np.std(success_speed_list) if success_speed_list else 0
    return_avg = np.mean(episode_return_list)
    return_std = np.std(episode_return_list)

    logging.info('%s: success rate: %.2f, collision rate: %.2f, timeout rate: %.2f, time: %.2f / %.2f, speed: %.2f / %.2f, return: %.3f / %.3f' %
                 (extra_str, success_rate, collision_rate, timeout_rate, success_time_avg, success_time_std, speed_avg, speed_std, return_avg, return_std))
    if phase == 'Test':
        logging.info('Collision episodes: %s', collision_episode_list)
        logging.info('Timeout episodes: %s', timeout_episode_list)
        logging.info('Computation time per step: %.3fs', np.mean(computation_time_list))
    return (success_rate, success_time_avg, return_avg)

def configure_args():
    parser = argparse.ArgumentParser(description='Test parameters')

    parser.add_argument('--policy_name', default='Ours')
    parser.add_argument('--model_path', default='output_dir/debug')
    parser.add_argument('--model_name', default='best_model.pt')
    parser.add_argument('--test_name', default='test_debug')
    parser.add_argument('--test_world', default='world.yaml')
    parser.add_argument('--robot_circle_num', type=int, default=5)      # 圆形机器人数量
    parser.add_argument('--robot_polygon_num', type=int, default=0)     # 多边形机器人数量
    parser.add_argument('--observation_radius', type=float, default=5)  # 观测半径/m
    parser.add_argument('--max_obstacles', type=int, default=5)         # 最大障碍数量
    parser.add_argument('--max_neighbors', type=int, default=5)         # 最大邻居数量
    parser.add_argument('--episode_max_steps', type=int, default=300)
    parser.add_argument('--val_test_episodes', type=int, default=100)
    parser.add_argument('--test_base_seed', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_figure', action='store_true')
    parser.add_argument('--save_gif', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--show_lidar', type=int, default=None)   # None: False; -1: All; >=0: robot_id
    parser.add_argument('--show_trajectory', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()

    return parser, args


if __name__ == '__main__':
    # Configure arguments
    parser, args = configure_args()

    # Configure model directory
    current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_path, args.model_path)

    # Load the hyperparameters from yaml
    with open(os.path.join(model_path, 'config.yaml'), 'r') as f:
        config = yaml.full_load(f)
    f.close()
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Configure the world file
    shutil.copyfile(os.path.join('env/worlds', args.test_world), os.path.join(model_path, args.test_name + '_world.yaml'))
    world_file = os.path.join(model_path, args.test_name + '_world.yaml')

    # Configure logging
    log_file = os.path.join(model_path, args.test_name + '.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler], format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Main
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logging.info('Using device: %s', device)
    logging.info('Using policy: %s', args.policy_name)
    logging.info('Using RL algorithm: %s', args.rl_algo_name)
    logging.info('Observation Coordination System: %s, Observation Type: %s, Spatio-Temporal Type: %s', args.obs_coord_system, args.obs_type, args.ST_type)
    
    env = gym.make('MultiRobotEnv-v0', policy_name=args.policy_name, world_file=world_file, robot_circle_num=args.robot_circle_num, robot_polygon_num=args.robot_polygon_num,
                   obs_coord_system=args.obs_coord_system, observation_radius=args.observation_radius, max_obstacles=args.max_obstacles, max_neighbors=args.max_neighbors,
                   sensor_obs_dim=args.sensor_obs_dim, agent_obs_dim=args.agent_obs_dim, costmap_obs_dim=args.costmap_obs_dim, obs_type=args.obs_type)
    logging.info('Square: %s, Circle: %s, Max Time: %.1fs, Step Time: %.1fs', env.square, env.circle, env.step_time * args.episode_max_steps, env.step_time)
    logging.info('Obstacle_circle_static Number:%2d, Obstacle_circle_dynamic Number:%2d', env.components['obstacles'].circle_static_num, env.components['obstacles'].circle_dynamic_num)
    logging.info('      Obstacle_polygon Number:%2d,           Obstacle_line Number:%2d', env.components['obstacles'].polygon_num, env.components['obstacles'].line_num)
    logging.info('          Robot_circle Number:%2d,           Robot_polygon Number:%2d', env.robot_circle_num, env.robot_polygon_num)
    logging.info('Robot Mode: %s, Min-Max Velocity: %s - %s', env.components['robots'].robot_mode, env.components['robots'].robot_vel_min, env.components['robots'].robot_vel_max)
    logging.info('Observation Radius: %.1f, Max Obstacles: %d, Max Neighbors: %d', env.observation_radius, env.max_obstacles, env.max_neighbors)
    logging.info('Robot Init Mode: %s, Random Yaw: %s, Random Radius: %s, Random Footprint: %s', env.components['robots'].robot_init_mode,
                 env.components['robots'].robot_random_yaw, env.components['robots'].robot_random_radius, env.components['robots'].robot_random_footprint)
    
    policy = PPOActorCritic(env, args.policy_name, args.ego_state_dim, args.sensor_obs_dim, args.agent_obs_dim, args.costmap_obs_dim,
                            args.obs_hn_dim, args.hidden_sizes, args.obs_type, args.ST_type, device)

    # Load the model
    end_state = torch.load(os.path.join(model_path, args.model_name))
    policy.load_state_dict(end_state['model'])
    logging.info('Loaded model from %s', args.model_name)

    test_loop(env, policy, 'Test', None, args.val_test_episodes, args.episode_max_steps, 0.99, None, args.test_base_seed, args.obs_coord_system,
              model_path, args.render, args.save_figure, args.save_gif, args.save_video, args.show_lidar, args.show_trajectory)