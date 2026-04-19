import numpy as np
from math import pi, cos, sin
from env.components.obstacle_circle import ObstacleCircle
from env.components.obstacle_line import ObstacleLine
from env.components.obstacle_polygon import ObstaclePolygon
from env.utils.collision_detection import collision_circle_matrix, collision_circle_circle, collision_circle_line, collision_circle_polygon
from env.utils.utils import distance

class EnvObstacles:
    def __init__(self, circle_static_num=0, circle_static_list=[], circle_dynamic_num=0, line_num=0, line_list=[], polygon_num=0, polygon_list=[], components=[], step_time=0.1, **kwargs):
        # kwargs: circle_init_mode, circle_radius_list, circle_vel_min, circle_vel_max, circle_start_list, circle_goal_list, circle_random_yaw, circle_random_radius, circle_interval, circle_task_interval
        self.circle_static_num = circle_static_num
        self.obstacle_circle_static_list = [ObstacleCircle(type='static', center=circle[0:2], radius=circle[2]) for circle in circle_static_list[0:circle_static_num]]
        self.line_num = line_num
        self.obstacle_line_list = [ObstacleLine(vertex_list) for vertex_list in line_list[0:line_num]]
        self.polygon_num = polygon_num
        self.obstacle_polygon_list = [ObstaclePolygon(vertex_list) for vertex_list in polygon_list[0:polygon_num]]

        self.circle_dynamic_num = circle_dynamic_num
        self.circle_init_mode = kwargs.get('circle_init_mode', 0)
        self.circle_vel_min = kwargs.get('circle_vel_min', -np.ones(2))
        self.circle_vel_max = kwargs.get('circle_vel_max', np.ones(2))
        self.circle_start_list = kwargs.get('circle_start_list', [])
        self.circle_goal_list = kwargs.get('circle_goal_list', [])
        self.circle_random_yaw = kwargs.get('circle_random_yaw', False)
        self.circle_random_radius = kwargs.get('circle_random_radius', False)
        self.circle_interval = kwargs.get('circle_interval', 1.0)
        self.circle_task_interval = kwargs.get('circle_task_interval', 5.0)
        self.components = components
        self.square = list(np.add(self.components['square'], [1, 1, -1, -1]))
        self.circle = list(np.add(self.components['circle'], [0, 0, -1]))
        
        if self.circle_dynamic_num > 0:
            circle_radius_list = kwargs['circle_radius_list'][:self.circle_dynamic_num]
            if len(circle_radius_list) < self.circle_dynamic_num:
                temp_radius = circle_radius_list[-1]
                circle_radius_list += [temp_radius for _ in range(self.circle_dynamic_num - len(circle_radius_list))]

        # dynamic circle obstacles
        self.obstacle_circle_dynamic_list = []
        for i in range(self.circle_dynamic_num):
            obstacle_circle = ObstacleCircle(id=i, type='dynamic', radius=circle_radius_list[i], vel_min=self.circle_vel_min, vel_max=self.circle_vel_max, step_time=step_time, **kwargs)
            self.obstacle_circle_dynamic_list.append(obstacle_circle)

    def obstacles_reset(self):
        # init_mode: 0: custom
        #            1: random
        #            2: circle_uniform
        #            3: circle_random
        # square area: x_min, y_min, x_max, y_max
        # circle area: x, y, radius

        if self.circle_dynamic_num == 0:
            return
        
        if self.circle_init_mode == 0:
            circle_start_list = [np.array(circle_start) for circle_start in self.circle_start_list]
            circle_goal_list = [np.array(circle_goal) for circle_goal in self.circle_goal_list]

        elif self.circle_init_mode == 1:
            circle_start_list, circle_goal_list = [], []
            while len(circle_start_list) < self.circle_dynamic_num:
                start = np.random.uniform(low=self.square[0:2]+[-pi], high=self.square[2:4]+[pi], size=3)
                if not self.reset_collision_check(start, circle_start_list, self.components, self.circle_interval):
                    goal = np.random.uniform(low=self.square[0:2]+[-pi], high=self.square[2:4]+[pi], size=3)
                    if not self.reset_collision_check(goal, circle_goal_list, self.components, self.circle_interval):
                        if distance(start[0:2], goal[0:2]) >= self.circle_task_interval:
                            goal[2] = start[2] = np.arctan2(goal[1]-start[1], goal[0]-start[0])
                            circle_start_list.append(start)
                            circle_goal_list.append(goal)

        elif self.circle_init_mode == 2:
            circle_start_list, circle_goal_list = [], []
            center = np.array(self.circle[0:2] + [0])
            theta = 0
            while len(circle_start_list) < self.circle_dynamic_num:
                start = center + np.array([cos(theta) * self.circle[2], sin(theta) * self.circle[2], theta + pi])
                goal = center + np.array([cos(theta+pi) * self.circle[2], sin(theta+pi) * self.circle[2], theta + pi])
                if not self.reset_collision_check(start, circle_start_list, self.components, 0):
                    circle_start_list.append(start)
                    circle_goal_list.append(goal)
                    theta += 2 * pi / self.circle_dynamic_num
        
        elif self.circle_init_mode == 3:
            circle_start_list, circle_goal_list = [], []
            center = np.array(self.circle[0:2] + [0])
            while len(circle_start_list) < self.circle_dynamic_num:
                theta = np.random.uniform(low=0, high=2*pi)
                start = center + np.array([cos(theta) * self.circle[2], sin(theta) * self.circle[2], theta + pi])
                goal = center + np.array([cos(theta+pi) * self.circle[2], sin(theta+pi) * self.circle[2], theta + pi])
                if not self.reset_collision_check(start, circle_start_list, self.components, self.circle_interval):
                    circle_start_list.append(start)
                    circle_goal_list.append(goal)
        
        if self.circle_random_yaw:
            for i in range(self.circle_dynamic_num):
                theta = np.random.uniform(low=-pi/2, high=pi/2)
                circle_start_list[i][2] = circle_start_list[i][2] + theta
                circle_goal_list[i][2] = circle_goal_list[i][2] + theta
        assert len(circle_start_list) == len(circle_goal_list) == self.circle_dynamic_num

        if self.circle_random_radius:
            for i in range(self.circle_dynamic_num):
                circle_radius = np.random.uniform(low=0.1, high=0.5)
                self.obstacle_circle_dynamic_list[i].radius = circle_radius

        for obstacle_circle, start, goal in zip(self.obstacle_circle_dynamic_list, circle_start_list, circle_goal_list):
            obstacle_circle.reset(start, goal)

    def reset_collision_check(self, check_pose, pose_list, components, circle_interval):
        self_circle = np.array([check_pose[0], check_pose[1], circle_interval/2])
 
        # check collision with map
        if components['map_matrix'] is not None:
            if collision_circle_matrix(self_circle, components['map_matrix'], components['map_origin'], components['map_resolution']):
                return True
        
        # check collision with static circle obstacles
        for obstacle_circle in self.obstacle_circle_static_list:
            temp_circle = np.array([obstacle_circle.state[0], obstacle_circle.state[1], obstacle_circle.radius])
            if collision_circle_circle(self_circle, temp_circle):
                return True

        # check collision with dynamic circle obstacles
        for pose in pose_list:
            if distance(check_pose[0:2], pose[0:2]) < circle_interval:
                return True
        
        # check collision with line obstacles
        for obstacle_line in self.obstacle_line_list:
            if collision_circle_line(self_circle, obstacle_line.segment):
                return True

        # check collision with polygon obstacles
        for obstacle_polygon in self.obstacle_polygon_list:
            if collision_circle_polygon(self_circle, obstacle_polygon):
                return True
            
        # check collision with robots
        for robot in components['robots'].robot_list:
            if robot.shape == 'circle':
                temp_circle = np.array([robot.state[0], robot.state[1], robot.radius])
                if collision_circle_circle(self_circle, temp_circle):
                    return True
            elif robot.shape == 'polygon':
                if collision_circle_polygon(self_circle, robot):
                     return True
                
        return False
    
    def reset_goal(self, obstacle_circle_id):
        old_goal_list = list(map(lambda a: a.goal, self.obstacle_circle_dynamic_list))
        while True:
            new_goal = np.random.uniform(low=self.square[0:2]+[-pi], high=self.square[2:4]+[pi], size=3)
            if not self.reset_collision_check(new_goal, old_goal_list, self.components, self.circle_interval):
                break
        self.obstacle_circle_dynamic_list[obstacle_circle_id].goal = new_goal

    def total_states(self):
        agent_state_list = list(map(lambda a: a.self_state(), self.obstacle_circle_dynamic_list))
        neighbor_state_list = list(map(lambda a: a.obs_state(), self.obstacle_circle_dynamic_list))
        obstacle_circle_static_state_list = list(map(lambda ocs: ocs.obs_state(), self.obstacle_circle_static_list))
        obstacle_line_state_list = list(map(lambda ol: ol.segment, self.obstacle_line_list))
        obstacle_polygon_state_list = list(map(lambda op: op.edge_list, self.obstacle_polygon_list))

        return [agent_state_list, neighbor_state_list, obstacle_circle_static_state_list, obstacle_line_state_list, obstacle_polygon_state_list]