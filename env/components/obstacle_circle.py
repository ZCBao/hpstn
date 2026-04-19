# import cvxpy
import numpy as np
from math import cos, sin, tan
from env.utils.kinematic_model import motion_abs
from env.utils.utils import distance, relative

class ObstacleCircle:
    def __init__(self, id=None, type='static', center=None, radius=0.2, vel_min=-np.ones(2), vel_max=np.ones(2), goal_tolerance=0.1, step_time=0.1, **kwargs):
        # kwargs: circle_radius_list, circle_vel_min, circle_vel_max, circle_start_list, circle_goal_list, circle_random_yaw, circle_random_radius, circle_interval
        if isinstance(vel_min, list):
            vel_min = np.array(vel_min)
        assert np.all(vel_min <= 0)
        if isinstance(vel_max, list):
            vel_max = np.array(vel_max)
        assert np.all(vel_max >= 0)
        
        self.id = id
        self.type = type    # static or dynamic
        self.center = center
        self.radius = radius
        self.vel_abs = np.zeros(2)

        if self.type == 'static':
            self.state = np.concatenate((center, [0]))
        elif self.type == 'dynamic':
            self.vel_min = vel_min
            self.vel_max = vel_max
            self.goal_tolerance = goal_tolerance
            self.arrive_flag = False
            self.step_time = step_time

            self.noise = kwargs.get('noise', False)

    def reset(self, start=np.zeros(3), goal=np.zeros(3)):
        self.state = start[:]
        self.start = start[:]
        self.goal = goal[:]
        self.vel_abs = np.zeros(2)
        self.arrive_flag = False

    def cal_des_vel_abs(self):
        dist, theta = relative(self.state, self.goal)

        if dist > self.goal_tolerance:
            if abs(tan(theta)) > self.vel_max[1] / self.vel_max[0]:
                speed = self.vel_max[1] / abs(sin(theta))
            else:
                speed = self.vel_max[0] / abs(cos(theta))
            vx = speed * cos(theta)
            vy = speed * sin(theta)
        else:
            vx = 0
            vy = 0

        return np.array([vx, vy])
        
    def move_forward(self, vel, stop=True):
        if stop:
            if self.arrive_flag:
                vel = np.zeros(2)

        self.vel_abs = np.clip(vel, self.vel_min, self.vel_max)
        self.state = motion_abs(self.state, self.vel_abs, self.step_time, self.noise)

    def arrive_check(self):
        dist = distance(self.state[0:2], self.goal[0:2])

        if dist < self.goal_tolerance:
            self.arrive_flag = True
            return True
        else:
            self.arrive_flag = False
            return False 
    
    def self_state(self):
        radius = self.radius * np.ones(1)
        v_des = self.cal_des_vel_abs()
        return np.concatenate((self.state[0:2], self.vel_abs, radius, v_des, self.goal[0:2]), axis=0)
    
    def obs_state(self):
        radius = self.radius * np.ones(1)
        return np.concatenate((self.state[0:2], self.vel_abs, radius), axis=0)