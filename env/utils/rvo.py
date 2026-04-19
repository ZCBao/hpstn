import numpy as np
from math import cos, sin, acos, asin, atan2, sqrt, pi, inf
from env.utils.utils import clip, cross, distance, vector_between_theta, wraptopi, circle_intersect_circle, line_in_circle, polygon_in_circle, \
                            circle2circle, circle2line, circle2polygon, polygon2circle, polygon2line, polygon2polygon, \
                            circle2circle_collision_time, circle2line_collision_time, circle2polygon_collision_time, \
                            polygon2circle_collision_time, polygon2line_collision_time, polygon2polygon_collision_time
# self_state: [px, py, vx, vy, Radius, vx_des, vy_des, gx, gy]
# neighbor_state_list: [[px, py, vx, vy, radius]]

class RVO:
    def __init__(self, obs_coord_system, observation_radius=5, max_neighbors=5, ctime_threshold=5):
        self.obs_coord_system = obs_coord_system
        self.observation_radius = observation_radius
        self.max_neighbors = max_neighbors
        self.ctime_threshold = ctime_threshold
    
    def config_vo_info(self, self_robot, robot_list, obstacle_circle_list=[], obstacle_line_list=[], obstacle_polygon_list=[], split_flag=False, global_flag=False):
        vo_info_list = []
        if self_robot.shape == 'circle':
            for robot in robot_list:
                if robot.id == self_robot.id:
                    continue
                if robot.shape == 'circle':
                    vo_info = self.vo_info_circle_circle(self_robot, robot, 'rvo', split_flag, global_flag)
                elif robot.shape == 'polygon':
                    vo_info = self.vo_info_circle_polygon(self_robot, robot, 'rvo', split_flag, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
            for obstacle_circle in obstacle_circle_list:
                vo_info = self.vo_info_circle_circle(self_robot, obstacle_circle, 'vo', True, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
            for obstacle_line in obstacle_line_list:
                vo_info = self.vo_info_circle_line(self_robot, obstacle_line, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
            for obstacle_polygon in obstacle_polygon_list:
                vo_info = self.vo_info_circle_polygon(self_robot, obstacle_polygon, 'vo', True, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
        elif self_robot.shape == 'polygon':
            for robot in robot_list:
                if robot.id == self_robot.id:
                    continue
                if robot.shape == 'circle':
                    vo_info = self.vo_info_polygon_circle(self_robot, robot, 'rvo', split_flag, global_flag)
                elif robot.shape == 'polygon':
                    vo_info = self.vo_info_polygon_polygon(self_robot, robot, 'rvo', split_flag, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
            for obstacle_circle in obstacle_circle_list:
                vo_info = self.vo_info_polygon_circle(self_robot, obstacle_circle, 'vo', True, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
            for obstacle_line in obstacle_line_list:
                vo_info = self.vo_info_polygon_line(self_robot, obstacle_line, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
            for obstacle_polygon in obstacle_polygon_list:
                vo_info = self.vo_info_polygon_polygon(self_robot, obstacle_polygon, 'vo', True, global_flag)
                if vo_info is not None:
                    vo_info_list.append(vo_info)
        
        vo_info_list.sort(key=lambda vo_info: (vo_info[2], vo_info[1]), reverse=True) # vo_info: [vo_obs, real_dist, ctime, vo_flag], sort in descending order of ctime and dist
        # vo_info_list.sort(key=lambda vo_info: vo_info[1], reverse=True) # vo_info: [vo_obs, real_dist, ctime, vo_flag], sort in descending order of dist
        if len(vo_info_list) > self.max_neighbors:
            vo_info_list_limited = vo_info_list[-self.max_neighbors:]
        else:
            vo_info_list_limited = vo_info_list
        if self.max_neighbors == 0:
            vo_info_list_limited = []

        obs_vo_list = []
        min_dist = inf
        min_ctime = inf
        vo_flag = False
        for vo_info in vo_info_list_limited:
            if self.obs_coord_system == 'global':
                obs_vo_list.append(vo_info[0])
            
            else:
                if self.obs_coord_system == 'local_yaw':
                    rot = self_robot.state[2]
                elif self.obs_coord_system == 'local_goal':
                    dgx = self_robot.goal[0] - self_robot.state[0]
                    dgy = self_robot.goal[1] - self_robot.state[1]
                    rot = atan2(dgy, dgx)

                voa_x =  vo_info[0][0] * cos(rot) + vo_info[0][1] * sin(rot)
                voa_y = -vo_info[0][0] * sin(rot) + vo_info[0][1] * cos(rot)
                vol_x =  vo_info[0][2] * cos(rot) + vo_info[0][3] * sin(rot)
                vol_y = -vo_info[0][2] * sin(rot) + vo_info[0][3] * cos(rot)
                vor_x =  vo_info[0][4] * cos(rot) + vo_info[0][5] * sin(rot)
                vor_y = -vo_info[0][4] * sin(rot) + vo_info[0][5] * cos(rot)
                real_dist = vo_info[0][6]
                input_ctime = vo_info[0][7]
                vo_obs_transformed = [voa_x, voa_y, vol_x, vol_y, vor_x, vor_y, real_dist, input_ctime]
                obs_vo_list.append(vo_obs_transformed)

            # if (vo_info[1] < min_dist and vo_info[3] == vo_flag) or (vo_info[3] > vo_flag):
            #     min_dist = vo_info[1]
            # if (vo_info[2] < min_ctime and vo_info[3] == vo_flag) or (vo_info[3] > vo_flag):
            #     min_ctime = vo_info[2]
            # if vo_info[3]:
            #     vo_flag = True
            if vo_info[1] < min_dist:
                min_dist = vo_info[1]
            if vo_info[2] < min_ctime:
                min_ctime = vo_info[2]
            if vo_info[3]:
                vo_flag = True

        return obs_vo_list, min_dist, min_ctime, vo_flag
    
    def vo_info_circle_circle(self, self_robot, circle, mode='vo', split_flag=False, global_flag=False):
        self_state = self_robot.self_state()
        circle_state = circle.obs_state()
        x, y, vx, vy, r = self_state[0:5]
        cx, cy, cvx, cvy, cr = circle_state[0:5]

        self_circle = np.array([x, y, r])
        circle_xyr = np.array([cx, cy, cr])

        dist = distance(self_state[0:2], circle_state[0:2])
        theta = atan2(cy - y, cx - x)
        if global_flag or (split_flag and dist <= self.observation_radius - cr) or (not split_flag and dist <= self.observation_radius):
            half_angle = asin(clip((r + cr)/dist, 0, 1))
            left_theta = wraptopi(theta + half_angle)
            right_theta = wraptopi(theta - half_angle)
        elif split_flag and dist < self.observation_radius + cr:
            region_circle = np.array([x, y, self.observation_radius])
            intersection_point1, intersection_point2 = circle_intersect_circle(region_circle, circle_xyr)
            intersection_point1_ = circle_state[0:2] + (intersection_point1 - circle_state[0:2]) * ((cr + r) / cr)
            intersection_point2_ = circle_state[0:2] + (intersection_point2 - circle_state[0:2]) * ((cr + r) / cr)
            tangent_line_len = sqrt(dist**2 - (cr+r)**2)
            intersection_line_len1 = distance(self_circle[0:2], intersection_point1_)
            intersection_line_len2 = distance(self_circle[0:2], intersection_point2_)
            if intersection_line_len1 >= tangent_line_len:
                half_angle1 = asin(clip((r + cr)/dist, 0, 1))
            else:
                half_angle1 = acos((dist**2 + intersection_line_len1**2 - (cr+r)**2) / (2 * dist * intersection_line_len1))
            if intersection_line_len2 >= tangent_line_len:
                half_angle2 = asin(clip((r + cr)/dist, 0, 1))
            else:
                half_angle2 = acos((dist**2 + intersection_line_len2**2 - (cr+r)**2) / (2 * dist * intersection_line_len2))
            left_theta = wraptopi(theta + half_angle1)
            right_theta = wraptopi(theta - half_angle2)
            half_angle = wraptopi((left_theta - right_theta) / 2)
        else:
            return None

        if mode == 'vo':
            apex = [cvx, cvy]
        elif mode == 'rvo':
            apex = [(vx + cvx)/2, (vy + cvy)/2]
        elif mode == 'hrvo':
            vo_apex = [cvx, cvy]
            rvo_apex = [(vx + cvx)/2, (vy + cvy)/2]
            dist_apex = distance(vo_apex, rvo_apex)
            theta_vo2rvo = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])
            if 2*half_angle == pi:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(0.001)
            else:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(2 * half_angle)
            centerline_vector = [cx - x, cy - y]
            diff_v_vector = [vx - rvo_apex[0], vy - rvo_apex[1]]
            if cross(centerline_vector, diff_v_vector) <= 0: 
                apex = [rvo_apex[0] - dist_diff * cos(right_theta), rvo_apex[1] - dist_diff * sin(right_theta)]
            else:
                apex = [vo_apex[0] + dist_diff * cos(right_theta), vo_apex[1] + dist_diff * sin(right_theta)]
            
        vo = apex + [left_theta, right_theta]
        real_dist = circle2circle(self_circle, circle_xyr)

        if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
            ctime = circle2circle_collision_time(self_circle, circle_xyr, vx-cvx, vy-cvy)
            if ctime < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                ctime = inf
        else:
            vo_flag = False
            ctime = inf
        input_ctime = 1 / (ctime + 0.2)

        vo_obs = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), real_dist, input_ctime]

        return vo_obs, real_dist, ctime, vo_flag
    
    def vo_info_circle_line(self, self_robot, line, global_flag=False):
        self_state = self_robot.self_state()
        x, y, vx, vy, r = self_state[0:5]
        lvx, lvy = 0, 0

        self_circle = np.array([x, y, r])
        if global_flag:
            line_vertex_list = line.vertex_list
        else:
            region_circle = np.array([x, y, self.observation_radius])
            line_vertex_list = line_in_circle(region_circle, line.segment)
            if not line_vertex_list:
                return None

        theta1 = atan2(line_vertex_list[0][1] - y, line_vertex_list[0][0] - x)
        theta2 = atan2(line_vertex_list[1][1] - y, line_vertex_list[1][0] - x)
        dist1 = distance([x, y], line_vertex_list[0])
        dist2 = distance([x, y], line_vertex_list[1])
        half_angle1 = asin(clip(r/dist1, 0, 1))
        half_angle2 = asin(clip(r/dist2, 0, 1))
        theta_list = [wraptopi(theta1 + half_angle1), wraptopi(theta1 - half_angle1), wraptopi(theta2 + half_angle2), wraptopi(theta2 - half_angle2)]
        theta_list.sort()
        max_angle = 0
        for i in range(len(theta_list)):
            angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
            if angle > max_angle:
                max_angle = angle
                if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                    left_theta = theta_list[i]
                    right_theta = theta_list[i-1]
                else:
                    left_theta = theta_list[i-1]
                    right_theta = theta_list[i]

        apex = [0, 0]
        vo = apex + [left_theta, right_theta]
        real_dist = circle2line(self_circle, line_vertex_list)

        if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
            ctime = circle2line_collision_time(self_circle, line_vertex_list, vx-lvx, vy-lvy)
            if ctime < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                ctime = inf
        else:
            vo_flag = False
            ctime = inf
        input_ctime = 1 / (ctime + 0.2)

        vo_obs = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), real_dist, input_ctime]

        return vo_obs, real_dist, ctime, vo_flag
    
    def vo_info_circle_polygon(self, self_robot, polygon, mode='vo', split_flag=False, global_flag=False):
        self_state = self_robot.self_state()
        polygon_state = polygon.obs_state()
        x, y, vx, vy, r = self_state[0:5]
        px, py, pvx, pvy, pr = polygon_state[0:5]

        self_circle = np.array([x, y, r])
        if global_flag or (not split_flag and distance(self_state, polygon_state) <= self.observation_radius):
            polygon_vertex_list = polygon.vertex_list
        elif split_flag:
            region_circle = np.array([x, y, self.observation_radius])
            polygon_vertex_list = polygon_in_circle(region_circle, polygon.edge_list)
            if not polygon_vertex_list:
                return None
        else:
            return None

        theta_list = []
        for i in range(len(polygon_vertex_list)):
            theta_j = atan2(polygon_vertex_list[i][1] - y, polygon_vertex_list[i][0] - x)
            dist_j = distance([x, y], polygon_vertex_list[i])
            half_angle_j = asin(clip(r/dist_j, 0, 1))
            theta_list.append(wraptopi(theta_j + half_angle_j))
            theta_list.append(wraptopi(theta_j - half_angle_j))
        theta_list.sort()
        max_angle = 0
        for i in range(len(theta_list)):
            angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
            if angle > max_angle:
                max_angle = angle
                if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                    left_theta = theta_list[i]
                    right_theta = theta_list[i-1]
                else:
                    left_theta = theta_list[i-1]
                    right_theta = theta_list[i]

        half_angle = wraptopi((left_theta - right_theta) / 2)
        
        if mode == 'vo':
            apex = [pvx, pvy]
        elif mode == 'rvo':
            apex = [(vx + pvx)/2, (vy + pvy)/2]
        elif mode == 'hrvo':
            vo_apex = [pvx, pvy]
            rvo_apex = [(vx + pvx)/2, (vy + pvy)/2]
            dist_apex = distance(vo_apex, rvo_apex)
            theta_vo2rvo = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])
            if 2*half_angle == pi:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(0.001)
            else:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(2 * half_angle)
            centerline_vector = [px - x, py - y]
            diff_v_vector = [vx - rvo_apex[0], vy - rvo_apex[1]]
            if cross(centerline_vector, diff_v_vector) <= 0: 
                apex = [rvo_apex[0] - dist_diff * cos(right_theta), rvo_apex[1] - dist_diff * sin(right_theta)]
            else:
                apex = [vo_apex[0] + dist_diff * cos(right_theta), vo_apex[1] + dist_diff * sin(right_theta)]
            
        vo = apex + [left_theta, right_theta]
        real_dist = circle2polygon(self_circle, polygon_vertex_list)

        if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
            ctime = circle2polygon_collision_time(self_circle, polygon_vertex_list, vx-pvx, vy-pvy)
            if ctime < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                ctime = inf
        else:
            vo_flag = False
            ctime = inf
        input_ctime = 1 / (ctime + 0.2)

        vo_obs = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), real_dist, input_ctime]

        return vo_obs, real_dist, ctime, vo_flag
    
    def vo_info_polygon_circle(self, self_robot, circle, mode='vo', split_flag=False, global_flag=False):
        self_state = self_robot.self_state()
        circle_state = circle.obs_state()
        x, y, vx, vy, r = self_state[0:5]
        cx, cy, cvx, cvy, cr = circle_state[0:5]

        self_vertex_list = self_robot.Vertex_list
        circle_xyr = np.array([cx, cy, cr])

        max_angle = 0
        theta_list = []
        dist = distance(self_state[0:2], circle_state[0:2])
        if global_flag or (split_flag and dist <= self.observation_radius - cr) or (not split_flag and dist <= self.observation_radius):
            for i in range(len(self_vertex_list)):
                theta_i = atan2(cy - self_vertex_list[i][1], cx - self_vertex_list[i][0])
                dist_i = distance(self_vertex_list[i], [cx, cy])
                half_angle_i = asin(clip(cr/dist_i, 0, 1))
                theta_list.append(wraptopi(theta_i + half_angle_i))
                theta_list.append(wraptopi(theta_i - half_angle_i))
            theta_list.sort()
            for i in range(len(theta_list)):
                angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
                if angle > max_angle:
                    max_angle = angle
                    if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                        left_theta = theta_list[i]
                        right_theta = theta_list[i-1]
                    else:
                        left_theta = theta_list[i-1]
                        right_theta = theta_list[i]
        elif split_flag and dist < self.observation_radius + cr:
            region_circle = np.array([x, y, self.observation_radius])
            intersection_point1, intersection_point2 = circle_intersect_circle(region_circle, circle_xyr)
            for i in range(len(self_vertex_list)):
                intersection_line_len_i1 = distance(self_vertex_list[i], intersection_point1)
                intersection_line_len_i2 = distance(self_vertex_list[i], intersection_point2)
                theta_i = atan2(cy - self_vertex_list[i][1], cx - self_vertex_list[i][0])
                dist_i = distance([cx, cy], self_vertex_list[i])
                tangent_line_len_i = sqrt(dist_i**2 - cr**2)
                if intersection_line_len_i1 >= tangent_line_len_i:
                    half_angle_i1 = asin(clip(cr/dist_i, 0, 1))
                else:
                    half_angle_i1 = acos((dist_i**2 + intersection_line_len_i1**2 - cr**2) / (2 * dist_i * intersection_line_len_i1))
                if intersection_line_len_i2 >= tangent_line_len_i:
                    half_angle_i2 = asin(clip(cr/dist_i, 0, 1))
                else:
                    half_angle_i2 = acos((dist_i**2 + intersection_line_len_i2**2 - cr**2) / (2 * dist_i * intersection_line_len_i2))
                theta_list.append(wraptopi(theta_i + half_angle_i1))
                theta_list.append(wraptopi(theta_i - half_angle_i2))
            theta_list.sort()
            for i in range(len(theta_list)):
                angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
                if angle > max_angle:
                    max_angle = angle
                    if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                        left_theta = theta_list[i]
                        right_theta = theta_list[i-1]
                    else:
                        left_theta = theta_list[i-1]
                        right_theta = theta_list[i]
        else:
            return None

        half_angle = wraptopi((left_theta - right_theta) / 2)

        if mode == 'vo':
            apex = [cvx, cvy]
        elif mode == 'rvo':
            apex = [(vx + cvx)/2, (vy + cvy)/2]
        elif mode == 'hrvo':
            vo_apex = [cvx, cvy]
            rvo_apex = [(vx + cvx)/2, (vy + cvy)/2]
            dist_apex = distance(vo_apex, rvo_apex)
            theta_vo2rvo = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])
            if 2*half_angle == pi:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(0.001)
            else:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(2 * half_angle)
            centerline_vector = [cx - x, cy - y]
            diff_v_vector = [vx - rvo_apex[0], vy - rvo_apex[1]]
            if cross(centerline_vector, diff_v_vector) <= 0: 
                apex = [rvo_apex[0] - dist_diff * cos(right_theta), rvo_apex[1] - dist_diff * sin(right_theta)]
            else:
                apex = [vo_apex[0] + dist_diff * cos(right_theta), vo_apex[1] + dist_diff * sin(right_theta)]
            
        vo = apex + [left_theta, right_theta]
        real_dist = polygon2circle(self_vertex_list, circle_xyr)

        if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
            ctime = polygon2circle_collision_time(self_vertex_list, circle_xyr, vx-cvx, vy-cvy)
            if ctime < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                ctime = inf
        else:
            vo_flag = False
            ctime = inf
        input_ctime = 1 / (ctime + 0.2)

        vo_obs = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), real_dist, input_ctime]

        return vo_obs, real_dist, ctime, vo_flag
    
    def vo_info_polygon_line(self, self_robot, line, global_flag=False):
        self_state = self_robot.self_state()
        x, y, vx, vy, r = self_state[0:5]
        lvx, lvy = 0, 0

        self_vertex_list = self_robot.Vertex_list
        if global_flag:
            line_vertex_list = line.vertex_list
        else:
            region_circle = np.array([x, y, self.observation_radius])
            line_vertex_list = line_in_circle(region_circle, line.segment)
            if not line_vertex_list:
                return None

        theta_list = []
        for self_vertex in self_vertex_list:
            for line_vertex in line_vertex_list:
                theta = atan2(line_vertex[1] - self_vertex[1], line_vertex[0] - self_vertex[0])
                theta_list.append(theta)
        theta_list.sort()
        max_angle = 0
        for i in range(len(theta_list)):
            angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
            if angle > max_angle:
                max_angle = angle
                if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                    left_theta = theta_list[i]
                    right_theta = theta_list[i-1]
                else:
                    left_theta = theta_list[i-1]
                    right_theta = theta_list[i]

        apex = [0, 0]
        vo = apex + [left_theta, right_theta]
        real_dist = polygon2line(self_vertex_list, line_vertex_list)

        if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
            ctime = polygon2line_collision_time(self_vertex_list, line_vertex_list, vx-lvx, vy-lvy)
            if ctime < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                ctime = inf
        else:
            vo_flag = False
            ctime = inf
        input_ctime = 1 / (ctime + 0.2)

        vo_obs = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), real_dist, input_ctime]

        return vo_obs, real_dist, ctime, vo_flag
    
    def vo_info_polygon_polygon(self, self_robot, polygon, mode='vo', split_flag=False, global_flag=False):
        self_state = self_robot.self_state()
        polygon_state = polygon.obs_state()
        x, y, vx, vy, r = self_state[0:5]
        px, py, pvx, pvy, pr = polygon_state[0:5]

        self_vertex_list = self_robot.Vertex_list
        if global_flag or (not split_flag and distance(self_state, polygon_state) <= self.observation_radius):
            polygon_vertex_list = polygon.vertex_list
        elif split_flag:
            region_circle = np.array([x, y, self.observation_radius])
            polygon_vertex_list = polygon_in_circle(region_circle, polygon.edge_list)
            if not polygon_vertex_list:
                return None
        else:
            return None

        theta_list = []
        for self_vertex in self_vertex_list:
            for polygon_vertex in polygon_vertex_list:
                theta = atan2(polygon_vertex[1] - self_vertex[1], polygon_vertex[0] - self_vertex[0])
                theta_list.append(theta)
        theta_list.sort()
        max_angle = 0
        for i in range(len(theta_list)):
            angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
            if angle > max_angle:
                max_angle = angle
                if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                    left_theta = theta_list[i]
                    right_theta = theta_list[i-1]
                else:
                    left_theta = theta_list[i-1]
                    right_theta = theta_list[i]

        half_angle = wraptopi((left_theta - right_theta) / 2)

        if mode == 'vo':
            apex = [pvx, pvy]
        elif mode == 'rvo':
            apex = [(vx + pvx)/2, (vy + pvy)/2]
        elif mode == 'hrvo':
            vo_apex = [pvx, pvy]
            rvo_apex = [(vx + pvx)/2, (vy + pvy)/2]
            dist_apex = distance(vo_apex, rvo_apex)
            theta_vo2rvo = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])
            if 2*half_angle == pi:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(0.001)
            else:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(2 * half_angle)
            centerline_vector = [px - x, py - y]
            diff_v_vector = [vx - rvo_apex[0], vy - rvo_apex[1]]
            if cross(centerline_vector, diff_v_vector) <= 0: 
                apex = [rvo_apex[0] - dist_diff * cos(right_theta), rvo_apex[1] - dist_diff * sin(right_theta)]
            else:
                apex = [vo_apex[0] + dist_diff * cos(right_theta), vo_apex[1] + dist_diff * sin(right_theta)]
            
        vo = apex + [left_theta, right_theta]
        real_dist = polygon2polygon(self_vertex_list, polygon_vertex_list)

        if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
            ctime = polygon2polygon_collision_time(self_vertex_list, polygon_vertex_list, vx-pvx, vy-pvy)
            if ctime < self.ctime_threshold:
                vo_flag = True
            else:
                vo_flag = False
                ctime = inf
        else:
            vo_flag = False
            ctime = inf
        input_ctime = 1 / (ctime + 0.2)

        vo_obs = [vo[0], vo[1], cos(vo[2]), sin(vo[2]), cos(vo[3]), sin(vo[3]), real_dist, input_ctime]

        return vo_obs, real_dist, ctime, vo_flag
    
    def preprocess(self, self_state, neighbor_state_list, obstacle_circle_state_list, obstacle_line_state_list, obstacle_polygon_state_list):
        # components in the observation radius
        ns_list = list(filter(lambda ns: 0 < distance(self_state, ns) <= self.observation_radius, neighbor_state_list))
        ocs_list = list(filter(lambda ocs: 0 < distance(self_state, ocs) <= self.observation_radius, obstacle_circle_state_list))
        region_circle = np.array([self_state[0], self_state[1], self.observation_radius])
        ols_list = list(map(lambda ols: line_in_circle(region_circle, ols), obstacle_line_state_list))
        ols_list = [ols for ols in ols_list if ols is not None]
        ops_list = list(map(lambda ops: polygon_in_circle(region_circle, ops), obstacle_polygon_state_list))
        ops_list = [ops for ops in ops_list if ops]

        return self_state, ns_list, ocs_list, ols_list, ops_list

    def cal_vo_vel(self, self_state, neighbor_state_list, obstacle_circle_static_state_list, obstacle_line_state_list, obstacle_polygon_state_list, vel_min=-np.ones(2), vel_max=np.ones(2)):
        self_state, ns_list, ocs_list, ols_list, ops_list = self.preprocess(self_state, neighbor_state_list, obstacle_circle_static_state_list, obstacle_line_state_list, obstacle_polygon_state_list)
        # mode: vo, rvo, hrvo
        vo_list1 = list(map(lambda x: self.vo_circle_circle(self_state, x, 'rvo'), ns_list))
        vo_list2 = list(map(lambda y: self.vo_circle_circle(self_state, y, 'vo'), ocs_list))
        vo_list3 = list(map(lambda z: self.vo_circle_line(self_state, z), ols_list))
        vo_list4 = list(map(lambda w: self.vo_circle_polygon(self_state, w), ops_list))
        vo_list = vo_list1 + vo_list2 + vo_list3 + vo_list4

        vo_outside, vo_inside = [], []
        for new_vx in np.arange(vel_min[0], vel_max[0], 0.05):
            for new_vy in np.arange(vel_min[1], vel_max[1], 0.05):
                if self.vel_out_vo_list(new_vx, new_vy, vo_list):
                    vo_outside.append([new_vx, new_vy])
                else:
                    vo_inside.append([new_vx, new_vy])
        
        vel_des = [self_state[5], self_state[6]]
        if (len(vo_outside) != 0):
            vo_vel = min(vo_outside, key = lambda v: distance(v, vel_des))
        else:
            vo_vel = min(vo_inside, key = lambda v: self.cal_penalty(v, vel_des, self_state, ns_list, ocs_list, ols_list, ops_list))

        return vo_vel

    def vo_circle_circle(self, self_state, neighbor_state, mode='rvo'):
        x, y, vx, vy, r = self_state[0:5]
        nx, ny, nvx, nvy, nr = neighbor_state[0:5]

        dist = distance(self_state[0:2], neighbor_state[0:2])
        theta = atan2(ny - y, nx - x)
        half_angle = asin(clip((r + nr)/dist, 0, 1))
        left_theta = wraptopi(theta + half_angle)
        right_theta = wraptopi(theta - half_angle)

        if mode == 'vo':
            apex = [nvx, nvy]
        elif mode == 'rvo':
            apex = [(vx + nvx)/2, (vy + nvy)/2]
        elif mode == 'hrvo':
            vo_apex = [nvx, nvy]
            rvo_apex = [(vx + nvx)/2, (vy + nvy)/2]
            dist_apex = distance(vo_apex, rvo_apex)
            theta_vo2rvo = atan2(rvo_apex[1] - vo_apex[1], rvo_apex[0] - vo_apex[0])
            if 2*half_angle == pi:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(0.001)
            else:
                dist_diff = dist_apex * sin(left_theta - theta_vo2rvo) / sin(2 * half_angle)
            centerline_vector = [nx - x, ny - y]
            diff_v_vector = [vx - rvo_apex[0], vy - rvo_apex[1]]
            if cross(centerline_vector, diff_v_vector) <= 0: 
                apex = [rvo_apex[0] - dist_diff * cos(right_theta), rvo_apex[1] - dist_diff * sin(right_theta)]
            else:
                apex = [vo_apex[0] + dist_diff * cos(right_theta), vo_apex[1] + dist_diff * sin(right_theta)]
        
        return apex + [left_theta, right_theta]

    def vo_circle_line(self, self_state, line_vertex_list):
        x, y, vx, vy, r = self_state[0:5]
        apex = [0, 0]

        theta1 = atan2(line_vertex_list[0][1] - y, line_vertex_list[0][0] - x)
        theta2 = atan2(line_vertex_list[1][1] - y, line_vertex_list[1][0] - x)
        dist1 = distance([x, y], line_vertex_list[0])
        dist2 = distance([x, y], line_vertex_list[1])
        half_angle1 = asin(clip(r/dist1, 0, 1))
        half_angle2 = asin(clip(r/dist2, 0, 1))
        theta_list = [wraptopi(theta1 + half_angle1), wraptopi(theta1 - half_angle1), wraptopi(theta2 + half_angle2), wraptopi(theta2 - half_angle2)]
        theta_list.sort()
        max_angle = 0
        for i in range(len(theta_list)):
            angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
            if angle > max_angle:
                max_angle = angle
                if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                    left_theta = theta_list[i]
                    right_theta = theta_list[i-1]
                else:
                    left_theta = theta_list[i-1]
                    right_theta = theta_list[i]

        return apex + [left_theta, right_theta]
    
    def vo_circle_polygon(self, self_state, polygon_vertex_list):
        x, y, vx, vy, r = self_state[0:5]
        apex = [0, 0]

        theta_list = []
        for i in range(len(polygon_vertex_list)):
            theta_j = atan2(polygon_vertex_list[i][1] - y, polygon_vertex_list[i][0] - x)
            dist_j = distance([x, y], polygon_vertex_list[i])
            half_angle_j = asin(clip(r/dist_j, 0, 1))
            theta_list.append(wraptopi(theta_j + half_angle_j))
            theta_list.append(wraptopi(theta_j - half_angle_j))
        theta_list.sort()
        max_angle = 0
        for i in range(len(theta_list)):
            angle = abs(wraptopi(theta_list[i] - theta_list[i-1]))
            if angle > max_angle:
                max_angle = angle
                if wraptopi(theta_list[i] - theta_list[i-1]) > 0:
                    left_theta = theta_list[i]
                    right_theta = theta_list[i-1]
                else:
                    left_theta = theta_list[i-1]
                    right_theta = theta_list[i]

        return apex + [left_theta, right_theta]
            
    def cal_penalty(self, vel, vel_des, self_state, neighbor_state_list, obstacle_circle_static_state_list, obstacle_line_state_list, obstacle_polygon_state_list, factor=1):
        tc_list = []
        self_circle = np.array([self_state[0], self_state[1], self_state[4]])

        for neighbor_state in neighbor_state_list:
            neighbor_circle = np.array([neighbor_state[0], neighbor_state[1], neighbor_state[4]])
            rel_vx = 2*vel[0] - neighbor_state[2] - self_state[2]
            rel_vy = 2*vel[1] - neighbor_state[3] - self_state[3]
            tc = circle2circle_collision_time(self_circle, neighbor_circle, rel_vx, rel_vy)
            tc_list.append(tc)

        for obstacle_circle in obstacle_circle_static_state_list:
            neighbor_circle = np.array([obstacle_circle[0], obstacle_circle[1], obstacle_circle[4]])
            rel_vx = vel[0] - obstacle_circle[2]
            rel_vy = vel[1] - obstacle_circle[3]
            tc = circle2circle_collision_time(self_circle, neighbor_circle, rel_vx, rel_vy)
            tc_list.append(tc)

        for obstacle_line in obstacle_line_state_list:
            tc = circle2line_collision_time(self_circle, obstacle_line, vel[0], vel[1])
            tc_list.append(tc)
        
        for obstacle_polygon in obstacle_polygon_state_list:
            tc = circle2polygon_collision_time(self_circle, obstacle_polygon, vel[0], vel[1])
            tc_list.append(tc)
   
        tc_min = min(tc_list)
        if tc_min == 0:
            tc_inv = inf
        else:
            tc_inv = 1/tc_min

        penalty = factor * tc_inv + distance(vel, vel_des)

        return penalty

    def vel_out_vo_list(self, vx, vy, vo_list):
        for vo in vo_list:
            if vector_between_theta([vx - vo[0], vy - vo[1]], vo[2], vo[3]):
                return False
        return True