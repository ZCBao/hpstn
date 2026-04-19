import numpy as np
import os
import torch
import yaml
from gym import spaces
from PIL import Image

from env.env_plot import EnvPlot
from env.env_robots import EnvRobots
from env.env_obstacles import EnvObstacles
from env.utils.lo import LO
from env.utils.rvo import RVO
from env.utils.utils import Timer, distance, norm, point2polygon, wraptopi

class EnvBase:
    def __init__(self, policy_name, world_file, obs_coord_system, observation_radius, max_obstacles, max_neighbors, sensor_obs_dim, agent_obs_dim, costmap_obs_dim, obs_type, **kwargs):
        # kwargs: robot_circle_num, robot_polygon_num
        self.timer = Timer()

        self.policy_name = policy_name
        self.obs_coord_system = obs_coord_system
        self.observation_radius = observation_radius
        self.max_obstacles = max_obstacles
        self.max_neighbors = max_neighbors
        self.sensor_obs_dim = sensor_obs_dim
        self.agent_obs_dim = agent_obs_dim
        self.costmap_obs_dim = costmap_obs_dim
        self.obs_type = obs_type

        self.components = dict()
        with open(world_file) as file:
            config_list = yaml.full_load(file)
            # world
            world_config = config_list['world']
            world_width = world_config.get('world_width', 10)
            world_height = world_config.get('world_height', 10)
            world_origin_x = world_config.get('world_origin_x', 0)
            world_origin_y = world_config.get('world_origin_y', 0)
            self.components['world_width'] = world_width
            self.components['world_height'] = world_height
            self.components['world_origin'] = np.array([world_origin_x, world_origin_y])
            world_map = world_config.get('world_map', None)
            if world_map is not None:
                world_map_file = os.path.join('env/maps', world_map)
                img = Image.open(world_map_file).convert('L')
                map_matrix = 255 - np.array(img)
                map_matrix = np.fliplr(map_matrix.T)
                map_origin_x = world_config.get('map_origin_x', 0)
                map_origin_y = world_config.get('map_origin_y', 0)
                map_resolution = world_config.get('map_resolution', 1)
                self.components['map_matrix'] = map_matrix
                self.components['map_origin'] = np.array([map_origin_x, map_origin_y])
                self.components['map_resolution'] = map_resolution
            else:
                self.components['map_matrix'] = None
            self.square = world_config.get('square', [0, 0, 10, 10])
            self.circle = world_config.get('circle', [5, 5, 5])
            self.components['square'] = self.square
            self.components['circle'] = self.circle
            self.step_time = world_config.get('step_time', 0.1)
            # robots
            self.robots_config = config_list.get('robots', dict())
            self.robot_circle_num = kwargs.get('robot_circle_num', self.robots_config.get('robot_circle_num', 0))
            self.robot_polygon_num = kwargs.get('robot_polygon_num', self.robots_config.get('robot_polygon_num', 0))
            self.robot_num = self.robot_circle_num + self.robot_polygon_num
            # obstacles_circle
            self.obstacles_circle_config = config_list.get('obstacles_circle', dict())
            # obstacles_line
            self.obstacles_line_config = config_list.get('obstacles_line', dict())
            # obstacles_polygon
            self.obstacles_polygon_config = config_list.get('obstacles_polygon', dict())
        
        self.components['obstacles'] = EnvObstacles(components=self.components, step_time=self.step_time,
                                                    **self.obstacles_circle_config, **self.obstacles_line_config, **self.obstacles_polygon_config)
        self.components['robots'] = EnvRobots(components=self.components, step_time=self.step_time, **self.robots_config, **kwargs)

        self.state_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([self.robots_config['robot_vel_min'][i] for i in range(len(self.robots_config['robot_vel_min']))], dtype=np.float32),
                                       high=np.array([self.robots_config['robot_vel_max'][i] for i in range(len(self.robots_config['robot_vel_max']))], dtype=np.float32))

        self.lo = LO(obs_coord_system, observation_radius, max_obstacles)
        self.rvo = RVO(obs_coord_system, observation_radius, max_neighbors)
        self.env_plot = EnvPlot(self.components)

    def env_observation(self):
        obs_list = []
        lo_dist_list, lo_ctime_list, lo_flag_list, vo_dist_list, vo_ctime_list, vo_flag_list = [], [], [], [], [], []
        for robot in self.components['robots'].robot_list:
            obs, lo_dist, lo_ctime, lo_flag, vo_dist, vo_ctime, vo_flag = self.robot_observation(robot)
            obs_list.append(obs)
            lo_dist_list.append(lo_dist)
            lo_ctime_list.append(lo_ctime)
            lo_flag_list.append(lo_flag)
            vo_dist_list.append(vo_dist)
            vo_ctime_list.append(vo_ctime)
            vo_flag_list.append(vo_flag)

        return obs_list, lo_dist_list, lo_ctime_list, lo_flag_list, vo_dist_list, vo_ctime_list, vo_flag_list

    def robot_observation(self, self_robot):
        self_state = self_robot.self_state()    # px, py, vx, vy, Radius, vx_des, vy_des, gx, gy
        robot_list = self.components['robots'].robot_list

        if self.obs_coord_system == 'local_yaw':
            rot = self_robot.state[2]
        elif self.obs_coord_system == 'local_goal':
            dgx = self_state[7] - self_state[0]
            dgy = self_state[8] - self_state[1]
            rot = np.arctan2(dgy, dgx)
        else:
            rot = 0.0
        
        vx = self_state[2] * np.cos(rot) + self_state[3] * np.sin(rot)
        vy = -self_state[2] * np.sin(rot) + self_state[3] * np.cos(rot)
        Radius = self_state[4]
        yaw = wraptopi(self_robot.state[2] - rot)
        # v_des_x = self_state[5] * np.cos(rot) + self_state[6] * np.sin(rot)
        # v_des_y = -self_state[5] * np.sin(rot) + self_state[6] * np.cos(rot)
        v_des = norm(self_state[5:7])
        dg = distance(self_state[0:2], self_state[7:9])
        # ego_state = torch.as_tensor([vx, vy, yaw, v_des, dg], dtype=torch.float32)
        # ego_state = torch.as_tensor([vx, vy, np.cos(yaw), np.sin(yaw), v_des, dg], dtype=torch.float32)
        ego_state = torch.as_tensor([vx, vy, Radius, yaw, v_des, dg], dtype=torch.float32)
        # ego_state = torch.as_tensor([vx, vy, Radius, v_des_x, v_des_y, dg], dtype=torch.float32)

        if 's' in self.obs_type and self_robot.lidar:
            self.timer.stop()
            self_robot.update_lidar_data(self.components)
            self.timer.restart()
            if isinstance(self.sensor_obs_dim, tuple):
                lo_dist = np.inf
                lo_ctime = np.inf
                lo_flag = False
                real_observation_radius = min(self.observation_radius, self_robot.lidar.range_max)
                for i, range_i in enumerate(self_robot.lidar.ranges):
                    if range_i < real_observation_radius:
                        theta = self_robot.state[2] + self_robot.lidar.install_pos[2] + self_robot.lidar.angles[i]
                        end_point = self_robot.lidar.start_point + [range_i * np.cos(theta), range_i * np.sin(theta)]
                        if self_robot.shape == 'circle':
                            dist = distance(self_state[0:2], end_point) - self_state[4]
                            dist = max(dist, 0)
                        elif self_robot.shape == 'polygon':
                            dist = point2polygon(end_point, self_robot.Vertex_list)
                        if dist < lo_dist:
                            lo_dist = dist
                sensor_obs = torch.as_tensor(self_robot.lidar.ranges / real_observation_radius, dtype=torch.float32).unsqueeze(0)
                if self.sensor_obs_dim[0] > 1:
                    last_sensor_obs = torch.as_tensor(self_robot.lidar.last_ranges / real_observation_radius, dtype=torch.float32).unsqueeze(0)
                    sensor_obs = torch.cat([last_sensor_obs, sensor_obs])
                    if self.sensor_obs_dim[0] > 2:
                        last_last_sensor_obs = torch.as_tensor(self_robot.lidar.last_last_ranges / real_observation_radius, dtype=torch.float32).unsqueeze(0)
                        sensor_obs = torch.cat([last_last_sensor_obs, sensor_obs])
                sensor_obs = torch.clamp(sensor_obs, 0.0, 1.0)
            elif isinstance(self.sensor_obs_dim, int):
                obs_lo_list, lo_dist, lo_ctime, lo_flag = self.lo.config_lo_info(self_robot)
                if len(obs_lo_list) == 0:
                    sensor_obs = torch.zeros((1, self.sensor_obs_dim))
                else:
                    sensor_obs = torch.as_tensor(obs_lo_list, dtype=torch.float32)[:, :self.sensor_obs_dim] # loa_x, loa_y, lol_x, lol_y, lor_x, lor_y, real_dist, input_ctime
        else:
            lo_dist = np.inf
            lo_ctime = np.inf
            lo_flag = False
            if isinstance(self.sensor_obs_dim, tuple):
                sensor_obs = torch.ones(self.sensor_obs_dim)
            elif isinstance(self.sensor_obs_dim, int):
                sensor_obs = torch.zeros((1, self.sensor_obs_dim))

        if 'a' in self.obs_type:
            obs_vo_list, vo_dist, vo_ctime, vo_flag = self.rvo.config_vo_info(self_robot, robot_list)
        else:
            obs_vo_list, vo_dist, vo_ctime, vo_flag = [], np.inf, np.inf, False
        if len(obs_vo_list) == 0:
            agent_obs = torch.zeros((1, self.agent_obs_dim))
        else:
            agent_obs = torch.as_tensor(obs_vo_list, dtype=torch.float32)[:, :self.agent_obs_dim] # voa_x, voa_y, vol_x, vol_y, vor_x, vor_y, real_dist, input_ctime
        
        observation = [ego_state, sensor_obs, agent_obs]

        return observation, lo_dist, lo_ctime, lo_flag, vo_dist, vo_ctime, vo_flag

    def cal_reward_done_info_list(self, lo_dist_list, lo_ctime_list, lo_flag_list, vo_dist_list, vo_ctime_list, vo_flag_list):
        reward_list, done_list, info_list = [], [], []
        for robot, lo_dist, lo_ctime, lo_flag, vo_dist, vo_ctime, vo_flag in zip(self.components['robots'].robot_list, lo_dist_list, lo_ctime_list, lo_flag_list, vo_dist_list, vo_ctime_list, vo_flag_list):
            reward, done, info = self.cal_reward_done_info(robot, lo_dist, lo_ctime, lo_flag, vo_dist, vo_ctime, vo_flag)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return reward_list, done_list, info_list

    def cal_reward_done_info(self, self_robot, lo_dist, lo_ctime, lo_flag, vo_dist, vo_ctime, vo_flag):
        if self_robot.collision_flag:
            collision_reward = -20
        else:
            min_dist = min(lo_dist, vo_dist)
            d_safe = 0.5
            k1 = 1.0
            k2 = 3.0
            if min_dist < d_safe:
                collision_reward = k1 * (np.tanh(k2*min_dist/d_safe)/np.tanh(k2) - 1)
            else:
                collision_reward = 0.0
        
        if self_robot.arrive_flag:
            goal_reward = 20
        else:
            k3 = 3.0
            delta_dist2goal = distance(self_robot.previous_state[0:2], self_robot.goal[0:2]) - distance(self_robot.state[0:2], self_robot.goal[0:2])
            previous_dir2goal = np.arctan2(self_robot.goal[1] - self_robot.previous_state[1], self_robot.goal[0] - self_robot.previous_state[0])
            current_dir2goal = np.arctan2(self_robot.goal[1] - self_robot.state[1], self_robot.goal[0] - self_robot.state[0])
            goal_reward = k3 * delta_dist2goal * np.cos(current_dir2goal - previous_dir2goal)
        
        reward = collision_reward + goal_reward
        # action_reward = -0.1 * abs(self_robot.vel_diff[1]) - 0.01
        # reward += action_reward

        if self_robot.done_flag:
            reward = 0
            if self_robot.collision_flag:
                info = -2
            elif self_robot.arrive_flag:
                info = 2
        else:
            if self_robot.collision_flag:
                self_robot.done_flag = True
                info = -1
            elif self_robot.arrive_flag:
                self_robot.done_flag = True
                info = 1
            else:
                info = 0
        
        return reward, self_robot.done_flag, info

    def robots_step(self, vel_list, stop=True):
        for i, robot in enumerate(self.components['robots'].robot_list):
            robot.move_forward(vel_list[i], stop)

        for robot in self.components['robots'].robot_list:
            robot.collision_check(self.components)
            robot.arrive_check()

    def obstacles_circle_step(self, stop=True):
        ts = self.components['obstacles'].total_states()
        vel_min = self.components['obstacles'].circle_vel_min
        vel_max = self.components['obstacles'].circle_vel_max
        vel_list = list(map(lambda agent_state: self.rvo.cal_vo_vel(agent_state, ts[1], ts[2], ts[3], ts[4], vel_min, vel_max), ts[0]))

        for i, obstacle_circle in enumerate(self.components['obstacles'].obstacle_circle_dynamic_list):
            obstacle_circle.move_forward(vel_list[i], stop)
            if obstacle_circle.arrive_check():
                self.components['obstacles'].reset_goal(obstacle_circle.id)
                # pass