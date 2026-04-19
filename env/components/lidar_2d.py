import numpy as np
from math import pi, inf, cos, sin
from env.utils.range_detection import range_line_matrix, range_line_circle, range_line_line

class Lidar2D:
    def __init__(self, install_pos=np.zeros(3), range_min=0, range_max=10, angle_min=-pi, angle_max=pi, range_num=180, **kwargs):
        self.id = None
        self.install_pos = install_pos
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.noise = kwargs.get('noise', False)
        self.std = 0.01

        self.ranges = np.full(range_num, inf)
        self.last_ranges = np.full(range_num, inf)
        self.last_last_ranges = np.full(range_num, inf)
        self.angles = np.linspace(self.angle_min, self.angle_max, num=range_num)
        self.start_point = np.zeros(2)
        self.end_points = np.zeros((range_num, 2))

    def update_data(self, self_state, components):
        self.last_last_ranges = self.last_ranges.copy()
        self.last_ranges = self.ranges.copy()
        start_point_x = self_state[0] + self.install_pos[0] * cos(self_state[2]) - self.install_pos[1] * sin(self_state[2])
        start_point_y = self_state[1] + self.install_pos[0] * sin(self_state[2]) + self.install_pos[1] * cos(self_state[2])
        self.start_point = np.array([start_point_x, start_point_y])

        thetas = self_state[2] + self.install_pos[2] + self.angles
        end_points = self.start_point + self.range_max * np.column_stack((np.cos(thetas), np.sin(thetas)))

        self.ranges = self.cal_ranges(end_points, components)

        if self.noise:
            self.ranges = np.clip(self.ranges, self.range_min, self.range_max)
            noises = np.random.normal(0, self.std, size=len(self.ranges))
            self.ranges = self.ranges + noises
            self.ranges = np.clip(self.ranges, self.range_min, self.range_max)
    
    def cal_ranges(self, end_points, components):
        ranges = np.full(len(self.angles), self.range_max)
        for i in range(len(self.angles)):
            self_line = np.array([[self.start_point[0], self.start_point[1]], [end_points[i][0], end_points[i][1]]])

            ray_min_x = min(self_line[0][0], self_line[1][0])
            ray_max_x = max(self_line[0][0], self_line[1][0])
            ray_min_y = min(self_line[0][1], self_line[1][1])
            ray_max_y = max(self_line[0][1], self_line[1][1])

            min_dist = inf
            min_intersection_point = self_line[1]
            collision_flag = False

            if components['map_matrix'] is not None:
                flag, intersection_point, dist = range_line_matrix(self_line, components['map_matrix'], components['map_origin'], components['map_resolution'])
                if flag and dist < min_dist:
                    min_dist = dist
                    min_intersection_point = intersection_point
                    collision_flag = True

            for obstacle_circle in components['obstacles'].obstacle_circle_static_list + components['obstacles'].obstacle_circle_dynamic_list:
                cx, cy, r = obstacle_circle.state[0], obstacle_circle.state[1], obstacle_circle.radius
                if (cx - r > ray_max_x) or (cx + r < ray_min_x) or (cy - r > ray_max_y) or (cy + r < ray_min_y):
                    continue
                temp_circle = np.array([cx, cy, r])
                flag, intersection_point, dist = range_line_circle(self_line, temp_circle)
                if flag and dist < min_dist:
                    min_dist = dist
                    min_intersection_point = intersection_point
                    collision_flag = True

            for obstacle_line in components['obstacles'].obstacle_line_list:
                segment = obstacle_line.segment
                segment_min_x = min(segment[0], segment[2])
                segment_max_x = max(segment[0], segment[2])
                segment_min_y = min(segment[1], segment[3])
                segment_max_y = max(segment[1], segment[3])
                if (segment_min_x > ray_max_x) or (segment_max_x < ray_min_x) or (segment_min_y > ray_max_y) or (segment_max_y < ray_min_y):
                    continue
                temp_line = np.array([[segment[0], segment[1]], [segment[2], segment[3]]])
                flag, intersection_point, dist = range_line_line(self_line, temp_line)
                if flag and dist < min_dist:
                    min_dist = dist
                    min_intersection_point = intersection_point
                    collision_flag = True

            for obstacle_polygon in components['obstacles'].obstacle_polygon_list:
                for edge in obstacle_polygon.edge_list:
                    segment_min_x = min(edge[0], edge[2])
                    segment_max_x = max(edge[0], edge[2])
                    segment_min_y = min(edge[1], edge[3])
                    segment_max_y = max(edge[1], edge[3])
                    if (segment_min_x > ray_max_x) or (segment_max_x < ray_min_x) or (segment_min_y > ray_max_y) or (segment_max_y < ray_min_y):
                        continue
                    temp_line = np.array([[edge[0], edge[1]], [edge[2], edge[3]]])
                    flag, intersection_point, dist = range_line_line(self_line, temp_line)
                    if flag and dist < min_dist:
                        min_dist = dist
                        min_intersection_point = intersection_point
                        collision_flag = True

            for robot in components['robots'].robot_list:
                if self.id == robot.id:
                    continue
                if robot.shape == 'circle':
                    cx, cy, r = robot.state[0], robot.state[1], robot.radius
                    if (cx - r > ray_max_x) or (cx + r < ray_min_x) or (cy - r > ray_max_y) or (cy + r < ray_min_y):
                        continue
                    temp_circle = np.array([cx, cy, r])
                    flag, intersection_point, dist = range_line_circle(self_line, temp_circle)
                    if flag and dist < min_dist:
                        min_dist = dist
                        min_intersection_point = intersection_point
                        collision_flag = True
                elif robot.shape == 'polygon':
                    for edge in robot.edge_list:
                        segment_min_x = min(edge[0], edge[2])
                        segment_max_x = max(edge[0], edge[2])
                        segment_min_y = min(edge[1], edge[3])
                        segment_max_y = max(edge[1], edge[3])
                        if (segment_min_x > ray_max_x) or (segment_max_x < ray_min_x) or (segment_min_y > ray_max_y) or (segment_max_y < ray_min_y):
                            continue
                        temp_line = np.array([[edge[0], edge[1]], [edge[2], edge[3]]])
                        flag, intersection_point, dist = range_line_line(self_line, temp_line)
                        if flag and dist < min_dist:
                            min_dist = dist
                            min_intersection_point = intersection_point
                            collision_flag = True

            if collision_flag and self.range_min <= min_dist < self.range_max:
                ranges[i] = min_dist
                self.end_points[i] = min_intersection_point
            else:
                ranges[i] = inf
                self.end_points[i] = min_intersection_point

        return ranges