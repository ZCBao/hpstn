import numpy as np
from math import cos, sin, asin, atan2, inf
from env.utils.utils import clip, distance, vector_between_theta, wraptopi, circle2polygon, polygon2polygon, circle2polygon_collision_time, polygon2polygon_collision_time
# self_state: [px, py, vx, vy, Radius, vx_des, vy_des, gx, gy]

class LO:
    def __init__(self, obs_coord_system, observation_radius=5, max_obstacles=5, ctime_threshold=5):
        self.obs_coord_system = obs_coord_system
        self.observation_radius = observation_radius
        self.max_obstacles = max_obstacles
        self.ctime_threshold = ctime_threshold
    
    def config_lo_info(self, self_robot):
        self_state = self_robot.self_state()
        x, y, vx, vy, r = self_state[0:5]
        yaw = self_robot.state[2]
        real_observation_radius = min(self.observation_radius, self_robot.lidar.range_max)
        
        lidar_angles = self_robot.lidar.angles
        lidar_ranges = self_robot.lidar.ranges
        lidar_range_num = len(self_robot.lidar.angles)
        start_i = None
        end_i = None
        lo_info_list = []
        for i in range(lidar_range_num):
            if lidar_ranges[i] < real_observation_radius:
                if start_i is None:
                    start_i = i
                    polygon_vertex_list = []
                if start_i is not None and i == lidar_range_num - 1:
                    end_i = i
                elif start_i is not None and abs(lidar_ranges[i] - lidar_ranges[i+1]) > 0.5:
                    end_i = i
                theta = yaw + self_robot.lidar.install_pos[2] + lidar_angles[i]
                end_point = self_robot.lidar.start_point + [lidar_ranges[i] * cos(theta), lidar_ranges[i] * sin(theta)]
                polygon_vertex_list.append(end_point)
            elif start_i is not None:
                end_i = i - 1
            
            if start_i is not None and end_i is not None and end_i >= start_i:
                if self_robot.shape == 'circle':
                    self_circle = np.array([x, y, r])
                    theta_list = []
                    for j in range(len(polygon_vertex_list)):
                        theta_j = atan2(polygon_vertex_list[j][1] - y, polygon_vertex_list[j][0] - x)
                        dist_j = distance([x, y], polygon_vertex_list[j])
                        half_angle_j = asin(clip(r/dist_j, 0, 1))
                        theta_list.append(wraptopi(theta_j + half_angle_j))
                        theta_list.append(wraptopi(theta_j - half_angle_j))
                    theta_list.sort()
                    max_angle = 0
                    for j in range(len(theta_list)):
                        angle = abs(wraptopi(theta_list[j] - theta_list[j-1]))
                        if angle > max_angle:
                            max_angle = angle
                            if wraptopi(theta_list[j] - theta_list[j-1]) > 0:
                                left_theta = theta_list[j]
                                right_theta = theta_list[j-1]
                            else:
                                left_theta = theta_list[j-1]
                                right_theta = theta_list[j]
                    
                    real_dist = circle2polygon(self_circle, polygon_vertex_list)
                    if vector_between_theta([vx, vy], left_theta, right_theta):
                        ctime = circle2polygon_collision_time(self_circle, polygon_vertex_list, vx, vy)
                        if ctime < self.ctime_threshold:
                            lo_flag = True
                        else:
                            lo_flag = False
                            ctime = inf
                    else:
                        lo_flag = False
                        ctime = inf
                
                elif self_robot.shape == 'polygon':
                    self_vertex_list = self_robot.Vertex_list
                    theta_list = []
                    for self_vertex in self_vertex_list:
                        for polygon_vertex in polygon_vertex_list:
                            theta = atan2(polygon_vertex[1] - self_vertex[1], polygon_vertex[0] - self_vertex[0])
                            theta_list.append(theta)
                    theta_list.sort()
                    max_angle = 0
                    for j in range(len(theta_list)):
                        angle = abs(wraptopi(theta_list[j] - theta_list[j-1]))
                        if angle > max_angle:
                            max_angle = angle
                            if wraptopi(theta_list[j] - theta_list[j-1]) > 0:
                                left_theta = theta_list[j]
                                right_theta = theta_list[j-1]
                            else:
                                left_theta = theta_list[j-1]
                                right_theta = theta_list[j]
                    
                    real_dist = polygon2polygon(self_vertex_list, polygon_vertex_list)
                    if vector_between_theta([vx, vy], left_theta, right_theta):
                        ctime = polygon2polygon_collision_time(self_vertex_list, polygon_vertex_list, vx, vy)
                        if ctime < self.ctime_threshold:
                            lo_flag = True
                        else:
                            lo_flag = False
                            ctime = inf
                    else:
                        lo_flag = False
                        ctime = inf

                apex = [self_robot.state[0], self_robot.state[1]]
                lo = apex + [left_theta, right_theta]
                input_ctime = 1 / (ctime + 0.2)
                lo_obs = [lo[0], lo[1], cos(lo[2]), sin(lo[2]), cos(lo[3]), sin(lo[3]), real_dist, input_ctime]
                
                lo_info = [lo_obs, real_dist, ctime, lo_flag]
                lo_info_list.append(lo_info)
                start_i = None
                end_i = None
        
        lo_info_list.sort(key=lambda lo_info: (lo_info[2], lo_info[1]), reverse=True) # lo_info: [lo_obs, real_dist, ctime, lo_flag], sort in descending order of ctime and dist
        if len(lo_info_list) > self.max_obstacles:
            lo_info_list_limited = lo_info_list[-self.max_obstacles:]
        else:
            lo_info_list_limited = lo_info_list
        if self.max_obstacles == 0:
            lo_info_list_limited = []

        obs_lo_list = []
        min_dist = inf
        min_ctime = inf
        lo_flag = False
        for lo_info in lo_info_list_limited:
            if self.obs_coord_system == 'global':
                obs_lo_list.append(lo_info[0])
            
            else:
                if self.obs_coord_system == 'local_yaw':
                    rot = self_robot.state[2]
                elif self.obs_coord_system == 'local_goal':
                    dgx = self_robot.goal[0] - self_robot.state[0]
                    dgy = self_robot.goal[1] - self_robot.state[1]
                    rot = atan2(dgy, dgx)
                
                loa_x =  (lo_info[0][0] - self_robot.state[0]) * cos(rot) + (lo_info[0][1] - self_robot.state[1]) * sin(rot)
                loa_y = -(lo_info[0][0] - self_robot.state[0]) * sin(rot) + (lo_info[0][1] - self_robot.state[1]) * cos(rot)
                lol_x =  lo_info[0][2] * cos(rot) + lo_info[0][3] * sin(rot)
                lol_y = -lo_info[0][2] * sin(rot) + lo_info[0][3] * cos(rot)
                lor_x =  lo_info[0][4] * cos(rot) + lo_info[0][5] * sin(rot)
                lor_y = -lo_info[0][4] * sin(rot) + lo_info[0][5] * cos(rot)
                real_dist = lo_info[0][6]
                input_ctime = lo_info[0][7]
                lo_obs_transformed = [loa_x, loa_y, lol_x, lol_y, lor_x, lor_y, real_dist, input_ctime]
                obs_lo_list.append(lo_obs_transformed)           
            
            # if (lo_info[1] < min_dist and lo_info[3] == lo_flag) or (lo_info[3] > lo_flag):
            #     min_dist = lo_info[1]
            # if (lo_info[2] < min_ctime and lo_info[3] == lo_flag) or (lo_info[3] > lo_flag):
            #     min_ctime = lo_info[2]
            # if lo_info[3]:
            #     lo_flag = True
            if lo_info[1] < min_dist:
                min_dist = lo_info[1]
            if lo_info[2] < min_ctime:
                min_ctime = lo_info[2]
            if lo_info[3]:
                lo_flag = True

        return obs_lo_list, min_dist, min_ctime, lo_flag