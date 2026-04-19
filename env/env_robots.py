import numpy as np
from math import pi, cos, sin
from env.components.robot import Robot
from env.utils.collision_detection import collision_circle_matrix, collision_circle_circle, collision_circle_line, collision_circle_polygon
from env.utils.utils import distance

class EnvRobots:
    def __init__(self, components=[], step_time=0.1, **kwargs):
        # kwargs: robot_circle_num, robot_polygon_num, robot_mode, robot_vel_min, robot_vel_max, robot_init_mode, robot_radius_list, robot_footprint_list,
        # robot_start_list, robot_goal_list, robot_random_yaw, robot_random_radius, robot_random_footprint, robot_interval, robot_task_interval
        self.robot_list = []
        self.robot_circle_num = kwargs.get('robot_circle_num', 0)
        self.robot_polygon_num = kwargs.get('robot_polygon_num', 0)
        self.robot_num = self.robot_circle_num + self.robot_polygon_num
        self.robot_mode = kwargs.get('robot_mode', 'diff')
        self.robot_vel_min = kwargs.get('robot_vel_min', -np.ones(2))
        self.robot_vel_max = kwargs.get('robot_vel_max', np.ones(2))
        self.robot_init_mode = kwargs.get('robot_init_mode', 0)
        self.robot_start_list = kwargs.get('robot_start_list', [])
        self.robot_goal_list = kwargs.get('robot_goal_list', [])
        self.robot_random_yaw = kwargs.get('robot_random_yaw', False)
        self.robot_random_radius = kwargs.get('robot_random_radius', False)
        self.robot_random_footprint = kwargs.get('robot_random_footprint', False)
        self.robot_interval = kwargs.get('robot_interval', 1.0)
        self.robot_task_interval = kwargs.get('robot_task_interval', 1.0)

        self.components = components
        self.square = self.components['square']
        self.circle = self.components['circle']

        if self.robot_circle_num > 0:
            robot_radius_list = kwargs['robot_radius_list'][:self.robot_circle_num]
            if len(robot_radius_list) < self.robot_circle_num:
                temp_radius = robot_radius_list[-1]
                robot_radius_list += [temp_radius for _ in range(self.robot_circle_num - len(robot_radius_list))]
        
        if self.robot_polygon_num > 0:
            robot_footprint_list = kwargs['robot_footprint_list'][:self.robot_polygon_num]
            if len(robot_footprint_list) < self.robot_polygon_num:
                temp_footprint = robot_footprint_list[-1]
                robot_footprint_list += [temp_footprint for _ in range(self.robot_polygon_num - len(robot_footprint_list))]

        # robots
        for i in range(self.robot_circle_num):
            robot = Robot(i, self.robot_mode, 'circle', radius=robot_radius_list[i], vel_min=self.robot_vel_min, vel_max=self.robot_vel_max, step_time=step_time, **kwargs)
            self.robot_list.append(robot)
        for i in range(self.robot_polygon_num):
            robot = Robot(self.robot_circle_num+i, self.robot_mode, 'polygon', footprint=robot_footprint_list[i], vel_min=self.robot_vel_min, vel_max=self.robot_vel_max, step_time=step_time, **kwargs)
            self.robot_list.append(robot)
        
    def robots_reset(self, safe_dist=0.0):
        # init_mode: 0: custom
        #            1: random
        #            2: circle_uniform
        #            3: circle_random
        #            4: passage_uniform
        #            5: passage_random
        # square area: x_min, y_min, x_max, y_max
        # circle area: x, y, radius

        if self.robot_init_mode == 0:
            robot_start_list = [np.array(robot_start) for robot_start in self.robot_start_list]
            robot_goal_list = [np.array(robot_goal) for robot_goal in self.robot_goal_list]
        
        elif self.robot_init_mode == 1:
            robot_start_list, robot_goal_list = [], []
            while len(robot_start_list) < self.robot_num:
                start = np.random.uniform(low=self.square[0:2]+[-pi], high=self.square[2:4]+[pi], size=3)
                if not self.reset_collision_check(start, robot_start_list, self.components, self.robot_interval):
                    goal = np.random.uniform(low=self.square[0:2]+[-pi], high=self.square[2:4]+[pi], size=3)
                    if not self.reset_collision_check(goal, robot_goal_list, self.components, self.robot_interval):
                        if distance(start[0:2], goal[0:2]) >= self.robot_task_interval:
                            goal[2] = start[2] = np.arctan2(goal[1]-start[1], goal[0]-start[0])
                            robot_start_list.append(start)
                            robot_goal_list.append(goal)
        
        elif self.robot_init_mode == 2:
            robot_start_list, robot_goal_list = [], []
            center = np.array(self.circle[0:2] + [0])
            theta = 0
            while len(robot_start_list) < self.robot_num:
                start = center + np.array([cos(theta) * self.circle[2], sin(theta) * self.circle[2], theta + pi])
                goal = center + np.array([cos(theta+pi) * self.circle[2], sin(theta+pi) * self.circle[2], theta + pi])
                if not self.reset_collision_check(start, robot_start_list, self.components, 0):
                    robot_start_list.append(start)
                    robot_goal_list.append(goal)
                    theta += 2 * pi / self.robot_num
        
        elif self.robot_init_mode == 3:
            robot_start_list, robot_goal_list = [], []
            center = np.array(self.circle[0:2] + [0])
            while len(robot_start_list) < self.robot_num:
                theta = np.random.uniform(low=0, high=2*pi)
                start = center + np.array([cos(theta) * self.circle[2], sin(theta) * self.circle[2], theta + pi])
                goal = center + np.array([cos(theta+pi) * self.circle[2], sin(theta+pi) * self.circle[2], theta + pi])
                if not self.reset_collision_check(start, robot_start_list, self.components, self.robot_interval):
                    robot_start_list.append(start)
                    robot_goal_list.append(goal)
        
        elif self.robot_init_mode == 4:
            robot_start_list, robot_goal_list = [], []
            num1 = self.robot_num // 2
            num2 = self.robot_num - num1
            if num1 > 0:
                step1 = (self.square[2] - self.square[0]) / (num1 + 1)
                start_list1 = [np.array([x, self.square[1], pi/2]) for x in np.arange(self.square[0]+step1, self.square[2], step1)][:num1]
                goal_list1 = [np.array([x, self.square[3], pi/2]) for x in np.arange(self.square[0]+step1, self.square[2], step1)][:num1]
                goal_list1.reverse()
                robot_start_list += start_list1
                robot_goal_list += goal_list1
            if num2 > 0:
                step2 = (self.square[2] - self.square[0]) / (num2 + 1)
                start_list2 = [np.array([x, self.square[3], -pi/2]) for x in np.arange(self.square[0]+step2, self.square[2], step2)][:num2]
                start_list2.reverse()
                goal_list2 = [np.array([x, self.square[1], -pi/2]) for x in np.arange(self.square[0]+step2, self.square[2], step2)][:num2]
                robot_start_list += start_list2
                robot_goal_list += goal_list2
        
        elif self.robot_init_mode == 5:
            robot_start_list, robot_goal_list = [], []
            num1 = self.robot_num // 2
            num2 = self.robot_num - num1
            while len(robot_start_list) < num1:
                start = np.random.uniform(low=[self.square[0], self.square[1], -pi], high=[self.square[2], (self.square[1]+self.square[3])/2, pi], size=3)
                if not self.reset_collision_check(start, robot_start_list, self.components, self.robot_interval):
                    goal = np.random.uniform(low=[self.square[0], (self.square[1]+self.square[3])/2, -pi], high=[self.square[2], self.square[3], pi], size=3)
                    if not self.reset_collision_check(goal, robot_goal_list, self.components, self.robot_interval):
                        if distance(start[0:2], goal[0:2]) >= self.robot_task_interval:
                            goal[2] = start[2] = np.arctan2(goal[1]-start[1], goal[0]-start[0])
                            robot_start_list.append(start)
                            robot_goal_list.append(goal)
            while len(robot_start_list) < num1 + num2:
                start = np.random.uniform(low=[self.square[0], (self.square[1]+self.square[3])/2, -pi], high=[self.square[2], self.square[3], pi], size=3)
                if not self.reset_collision_check(start, robot_start_list, self.components, self.robot_interval):
                    goal = np.random.uniform(low=[self.square[0], self.square[1], -pi], high=[self.square[2], (self.square[1]+self.square[3])/2, pi], size=3)
                    if not self.reset_collision_check(goal, robot_goal_list, self.components, self.robot_interval):
                        if distance(start[0:2], goal[0:2]) >= self.robot_task_interval:
                            goal[2] = start[2] = np.arctan2(goal[1]-start[1], goal[0]-start[0])
                            robot_start_list.append(start)
                            robot_goal_list.append(goal)
        
        if self.robot_random_yaw:
            for i in range(self.robot_num):
                theta = np.random.uniform(low=-pi, high=pi)
                robot_start_list[i][2] = robot_start_list[i][2] + theta
                robot_goal_list[i][2] = robot_goal_list[i][2] + theta
        assert len(robot_start_list) == len(robot_goal_list) == self.robot_num

        if self.robot_random_radius:
            for i in range(self.robot_circle_num):
                robot_radius = np.random.uniform(low=0.1, high=0.5)
                self.robot_list[i].radius = robot_radius
        if self.robot_random_footprint:
            for i in range(self.robot_polygon_num):
                robot_footprint_xy = np.random.uniform(low=[0.1, 0.1], high=[0.5, 0.5])
                self.robot_list[self.robot_circle_num+i].footprint = np.array([robot_footprint_xy[0], robot_footprint_xy[0], robot_footprint_xy[1], robot_footprint_xy[1]])

        for robot, start, goal in zip(self.robot_list, robot_start_list, robot_goal_list):
            robot.reset(start, goal, safe_dist)

    def reset_collision_check(self, check_pose, pose_list, components, robot_interval):
        self_circle = np.array([check_pose[0], check_pose[1], robot_interval/2])
        
        # check collision with map
        if components['map_matrix'] is not None:
            if collision_circle_matrix(self_circle, components['map_matrix'], components['map_origin'], components['map_resolution']):
                return True
        
        # check collision with circle obstacles
        for obstacle_circle in components['obstacles'].obstacle_circle_static_list:
            temp_circle = np.array([obstacle_circle.state[0], obstacle_circle.state[1], obstacle_circle.radius])
            if collision_circle_circle(self_circle, temp_circle):
                return True
        
        # check collision with line obstacles
        for obstacle_line in components['obstacles'].obstacle_line_list:
            if collision_circle_line(self_circle, obstacle_line.segment):
                return True
        
        # check collision with polygon obstacles
        for obstacle_polygon in components['obstacles'].obstacle_polygon_list:
            if collision_circle_polygon(self_circle, obstacle_polygon):
                return True

        # check collision with robots
        for pose in pose_list:
            if distance(check_pose[0:2], pose[0:2]) < robot_interval:
                return True
                
        return False