import numpy as np
from collections import namedtuple
from math import cos, sin, atan2, pi
from env.components.lidar_2d import Lidar2D
from env.utils.collision_detection import collision_circle_matrix, collision_circle_circle, collision_circle_line, collision_circle_polygon, \
                                          collision_polygon_matrix, collision_polygon_circle, collision_polygon_line, collision_polygon_polygon
from env.utils.kinematic_model import motion_diff, motion_omni, motion_ros
from env.utils.utils import distance, norm, relative, wraptopi

class Robot:
    def __init__(self, id, mode='diff', shape='circle', radius=0, footprint=[], vel_min=-np.ones(2), vel_max=np.ones(2), goal_tolerance=0.1, step_time=0.1, **kwargs):
        # kwargs: robot_circle_num, robot_polygon_num, robot_mode, robot_vel_min, robot_vel_max, robot_init_mode, robot_radius_list, robot_footprint_list,
        # robot_start_list, robot_goal_list, robot_random_yaw, robot_random_radius, robot_random_footprint, robot_interval, robot_task_interval
        if isinstance(vel_min, list):
            vel_min = np.array(vel_min)
        assert np.all(vel_min <= 0)
        if isinstance(vel_max, list):
            vel_max = np.array(vel_max)
        assert np.all(vel_max >= 0)
        
        self.id = int(id)
        self.mode = mode
        self.shape = shape
        if self.shape == 'circle':
            self.radius = radius
        elif self.shape == 'polygon':
            assert len(footprint) == 4
            self.footprint = np.array(footprint)
        self.vel_abs = np.zeros(3)
        if self.mode == 'diff':
            self.vel_diff = np.zeros(2)
        elif self.mode == 'omni':
            self.vel_omni = np.zeros(2)
        elif self.mode == 'ros':
            self.vel_ros = np.zeros(3)
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.goal_tolerance = goal_tolerance
        self.arrive_flag = False
        self.collision_flag = False
        self.done_flag = False

        self.step_time = step_time

        lidar_config = kwargs.get('lidar2d', None)
        if lidar_config is not None:
            self.lidar = Lidar2D(**lidar_config)
            self.lidar.id = self.id
        else:
            self.lidar = None
        
        self.noise = kwargs.get('noise', False)
    
    def reset(self, start=np.zeros(3), goal=np.zeros(3), safe_dist=0.0):
        start[2] = wraptopi(start[2])
        goal[2] = wraptopi(goal[2])

        self.start = start[:]
        self.goal = goal[:]
        self.state = start[:]   # [x, y, yaw]
        self.previous_state = start[:]

        self.vel_abs = np.zeros(3)
        if self.mode == 'diff':
            self.vel_diff = np.zeros(2)
        elif self.mode == 'omni':
            self.vel_omni = np.zeros(2)
        elif self.mode == 'ros':
            self.vel_ros = np.zeros(3)
        
        self.arrive_flag = False
        self.collision_flag = False
        self.done_flag = False
        
        if self.shape == 'circle':
            self.Radius = self.radius + safe_dist
        elif self.shape == 'polygon':
            self.radius = np.sqrt(np.max(self.footprint[0:2])**2 + np.max(self.footprint[2:4])**2)
            self.Footprint = self.footprint + safe_dist
            self.Radius = np.sqrt(np.max(self.Footprint[0:2])**2 + np.max(self.Footprint[2:4])**2)
            self.vertex_list, self.edge_list = self.cal_polygon_params(self.footprint)
            self.Vertex_list, self.Edge_list = self.cal_polygon_params(self.Footprint)

    def cal_polygon_params(self, footprint):
        front_left_x = self.state[0] + footprint[0] * cos(self.state[2]) - footprint[2] * sin(self.state[2])
        front_left_y = self.state[1] + footprint[0] * sin(self.state[2]) + footprint[2] * cos(self.state[2])
        rear_left_x = self.state[0] - footprint[1] * cos(self.state[2]) - footprint[2] * sin(self.state[2])
        rear_left_y = self.state[1] - footprint[1] * sin(self.state[2]) + footprint[2] * cos(self.state[2])
        rear_right_x = self.state[0] - footprint[1] * cos(self.state[2]) + footprint[3] * sin(self.state[2])
        rear_right_y = self.state[1] - footprint[1] * sin(self.state[2]) - footprint[3] * cos(self.state[2])
        front_right_x = self.state[0] + footprint[0] * cos(self.state[2]) + footprint[3] * sin(self.state[2])
        front_right_y = self.state[1] + footprint[0] * sin(self.state[2]) - footprint[3] * cos(self.state[2])
        vertex_list = [np.array([front_left_x, front_left_y]), np.array([rear_left_x, rear_left_y]), np.array([rear_right_x, rear_right_y]), np.array([front_right_x, front_right_y])]
        edge_list = [np.concatenate([vertex_list[i], vertex_list[(i+1)%len(vertex_list)]]) for i in range(len(vertex_list))]

        return vertex_list, edge_list
    
    def update_lidar_data(self, components):
        if self.lidar is not None:
            self.lidar.update_data(self.state, components)

    def cal_des_vel_abs(self):
        dist, theta = relative(self.state, self.goal)
        if dist > self.goal_tolerance:
            if self.mode == 'diff':
                vx = self.vel_max[0] * cos(theta)
                vy = self.vel_max[0] * sin(theta)
            elif self.mode == 'omni' or self.mode == 'ros':
                max_speed = np.max(self.vel_max[0:2])
                vx = max_speed * cos(theta)
                vy = max_speed * sin(theta)
        else:
            vx = 0
            vy = 0
        
        return np.array([vx, vy])
    
    def move_forward(self, vel, stop=True):
        # mode: diff: np.array([v, w])
        #       omni: np.array([vx, vy])
        #        ros: np.array([vx, vy, wz])
        if stop:
            if self.arrive_flag or self.collision_flag:
                vel = np.zeros(np.shape(vel))

        self.previous_state = self.state[:]

        if self.mode == 'diff':
            self.vel_diff = self.vel_abs2diff(vel)
            self.vel_abs = self.vel_diff2abs(self.vel_diff)
            self.state = motion_diff(self.state, self.vel_diff, self.step_time, self.noise)
        elif self.mode == 'omni':
            self.vel_omni = self.vel_abs2omni(vel)
            self.vel_abs = self.vel_omni2abs(self.vel_omni)
            self.state = motion_omni(self.state, self.vel_omni, self.step_time, self.noise)
        elif self.mode == 'ros':
            self.vel_ros = self.vel_abs2ros(vel)
            self.vel_abs = self.vel_ros2abs(self.vel_ros)
            self.state = motion_ros(self.state, self.vel_ros, self.step_time, self.noise)
        
        if self.shape == 'polygon':
            self.vertex_list, self.edge_list = self.cal_polygon_params(self.footprint)
            self.Vertex_list, self.Edge_list = self.cal_polygon_params(self.Footprint)

    def vel_abs2diff(self, vel_abs):
        yaw = self.state[2]
        speed = norm(vel_abs[0:2])
        if speed < 1e-6:
            vel_diff = np.array([0.0, 0.0])
        else:
            speed_theta = atan2(vel_abs[1], vel_abs[0])
            diff_theta = wraptopi(speed_theta - yaw)
            v = speed * cos(diff_theta)
            w = diff_theta / self.step_time
            # if abs(diff_theta) < 1e-6 or abs(wraptopi(diff_theta - pi)) < 1e-6:
            #     v = vel_abs[0] * cos(yaw) + vel_abs[1] * sin(yaw)
            #     w = 0
            # else:
            #     w = 2 * diff_theta / self.step_time
            #     v = speed * diff_theta / sin(diff_theta) * np.sign(vel_abs[0] * cos(yaw) + vel_abs[1] * sin(yaw))
            v_limited = np.clip(v, self.vel_min[0], self.vel_max[0])
            w_limited = np.clip(w, self.vel_min[1], self.vel_max[1])
            vel_diff = np.array([v_limited, w_limited])

        return vel_diff
    
    def vel_diff2abs(self, vel_diff):
        yaw = self.state[2]
        if abs(vel_diff[1]) < 1e-6:
            vx = vel_diff[0] * cos(yaw)
            vy = vel_diff[0] * sin(yaw)
        else:
            ratio = vel_diff[0] / vel_diff[1]
            vx = ( ratio * sin(yaw + vel_diff[1] * self.step_time) - ratio * sin(yaw)) / self.step_time
            vy = (-ratio * cos(yaw + vel_diff[1] * self.step_time) + ratio * cos(yaw)) / self.step_time
        vel_abs = np.array([vx, vy, vel_diff[1]])

        return vel_abs

    def vel_abs2omni(self, vel_abs):
        yaw = self.state[2]
        v_x = vel_abs[0] * cos(yaw) + vel_abs[1] * sin(yaw)
        v_y = -vel_abs[0] * sin(yaw) + vel_abs[1] * cos(yaw)
        vel_omni = np.clip(np.array([v_x, v_y]), self.vel_min, self.vel_max)

        return vel_omni
    
    def vel_omni2abs(self, vel_omni):
        yaw = self.state[2]
        vx = vel_omni[0] * cos(yaw) - vel_omni[1] * sin(yaw)
        vy = vel_omni[0] * sin(yaw) + vel_omni[1] * cos(yaw)
        vel_abs = np.array([vx, vy, 0.0])

        return vel_abs

    def vel_abs2ros(self, vel_abs):
        yaw = self.state[2]
        v_x = vel_abs[0] * cos(yaw) + vel_abs[1] * sin(yaw)
        v_y = -vel_abs[0] * sin(yaw) + vel_abs[1] * cos(yaw)
        w_z = vel_abs[2]
        vel_ros = np.clip(np.array([v_x, v_y, w_z]), self.vel_min, self.vel_max)

        return vel_ros
    
    def vel_ros2abs(self, vel_ros):
        yaw = self.state[2]

        # tangent line model
        vx = vel_ros[0] * cos(yaw) - vel_ros[1] * sin(yaw)
        vy = vel_ros[0] * sin(yaw) + vel_ros[1] * cos(yaw)
        vel_abs = np.array([vx, vy, vel_ros[2]])

        # # cut line model
        # v = norm(vel_ros[0:2])
        # direction = yaw + vel_ros[2] * self.step_time / 2
        # vx = v * cos(direction)
        # vy = v * sin(direction)
        # vel_abs = np.array([vx, vy, vel_ros[2]])

        # # arc model
        # v = norm(vel_ros[0:2])
        # w = vel_ros[2]
        # if abs(w) < 1e-6:
        #     ratio = v / w
        #     vx = (ratio * sin(yaw + w * self.step_time) - ratio * sin(yaw)) / self.step_time
        #     vy = (-ratio * cos(yaw + w * self.step_time) + ratio * cos(yaw)) / self.step_time
        #     vel_abs = np.array([vx, vy, w])
        # else:
        #     vx = v * cos(yaw)
        #     vy = v * sin(yaw)
        #     vel_abs = np.array([vx, vy, w])

        return vel_abs

    def arrive_check(self):
        if self.arrive_flag:
            return True

        dist = distance(self.state[0:2], self.goal[0:2])

        if dist < self.goal_tolerance:
            self.arrive_flag = True
            return True
        else:
            self.arrive_flag = False
            return False

    def collision_check(self, components):
        if self.collision_flag == True:
            return True
        
        if self.shape == 'circle':
            self_circle = np.array([self.state[0], self.state[1], self.Radius])

            # check collision with map
            if components['map_matrix'] is not None:
                if collision_circle_matrix(self_circle, components['map_matrix'], components['map_origin'], components['map_resolution']):
                    self.collision_flag = True
                    return True
        
            # check collision with circle obstacles
            for obstacle_circle in components['obstacles'].obstacle_circle_static_list + components['obstacles'].obstacle_circle_dynamic_list:
                temp_circle = np.array([obstacle_circle.state[0], obstacle_circle.state[1], obstacle_circle.radius])
                if collision_circle_circle(self_circle, temp_circle):
                    self.collision_flag = True
                    return True

            # check collision with line obstacles
            for obstacle_line in components['obstacles'].obstacle_line_list:
                if collision_circle_line(self_circle, obstacle_line.segment):
                    self.collision_flag = True
                    return True

            # check collision with polygon obstacles
            for obstacle_polygon in components['obstacles'].obstacle_polygon_list:
                if collision_circle_polygon(self_circle, obstacle_polygon):
                    self.collision_flag = True
                    return True
        
            # check collision with robots
            for robot in components['robots'].robot_list:
                if robot.id != self.id:
                    if robot.shape == 'circle':
                        temp_circle = np.array([robot.state[0], robot.state[1], robot.radius])
                        if collision_circle_circle(self_circle, temp_circle):
                            robot.collision_flag = True
                            self.collision_flag = True
                            return True
                    elif robot.shape == 'polygon':
                        if collision_circle_polygon(self_circle, robot):
                            robot.collision_flag = True
                            self.collision_flag = True
                            return True
        
        elif self.shape == 'polygon':
            polygon = namedtuple('polygon', ['vertex_list', 'edge_list'])
            self_polygon = polygon(self.Vertex_list, self.Edge_list)

            # check collision with map
            if components['map_matrix'] is not None:
                if collision_polygon_matrix(self_polygon, components['map_matrix'], components['map_origin'], components['map_resolution']):
                    self.collision_flag = True
                    return True
            
            # check collision with circle obstacles
            for obstacle_circle in components['obstacles'].obstacle_circle_static_list + components['obstacles'].obstacle_circle_dynamic_list:
                temp_circle = np.array([obstacle_circle.state[0], obstacle_circle.state[1], obstacle_circle.radius])
                if collision_polygon_circle(self_polygon, temp_circle):
                    self.collision_flag = True
                    return True
                
            # check collision with line obstacles
            for obstacle_line in components['obstacles'].obstacle_line_list:
                if collision_polygon_line(self_polygon, obstacle_line.segment):
                    self.collision_flag = True
                    return True
            
            # check collision with polygon obstacles
            for obstacle_polygon in components['obstacles'].obstacle_polygon_list:
                if collision_polygon_polygon(self_polygon, obstacle_polygon):
                    self.collision_flag = True
                    return True
            
            # check collision with robots
            for robot in components['robots'].robot_list:
                if robot.id != self.id:
                    if robot.shape == 'circle':
                        temp_circle = np.array([robot.state[0], robot.state[1], robot.radius])
                        if collision_polygon_circle(self_polygon, temp_circle):
                            robot.collision_flag = True
                            self.collision_flag = True
                            return True
                    elif robot.shape == 'polygon':
                        if collision_polygon_polygon(self_polygon, robot):
                            robot.collision_flag = True
                            self.collision_flag = True
                            return True
        
        return False

    def self_state(self):
        # px, py, vx, vy, Radius, vx_des, vy_des, gx, gy
        Radius = self.Radius * np.ones(1)
        v_des = self.cal_des_vel_abs()
        return np.concatenate((self.state[0:2], self.vel_abs[0:2], Radius, v_des, self.goal[0:2]), axis=0)

    def obs_state(self):
        # px, py, vx, vy, radius
        radius = self.radius * np.ones(1)
        return np.concatenate((self.state[0:2], self.vel_abs[0:2], radius), axis=0)