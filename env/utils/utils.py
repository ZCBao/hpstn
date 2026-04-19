import numpy as np
import time
from math import pi, inf, cos, sin, acos, atan2, sqrt

class Timer:
    def __init__(self):
        self._total = 0.0
        self._start = None
        self._running = False
    def start(self):
        if not self._running:
            self._start = time.perf_counter()
            self._running = True
        else:
            raise RuntimeError("Timer is already running!")
    def stop(self):
        if self._running:
            self._total += time.perf_counter() - self._start
            self._start = None
    def restart(self):
        if self._running:
            self._start = time.perf_counter()
    def end(self):
        if self._running:
            self._total += time.perf_counter() - self._start
            total = self._total
            self._start = None
            self._total = 0.0
            self._running = False
            return total
        else:
            raise RuntimeError("Timer is not running!")

def clip(x, min_x, max_x):
    return max(min(max_x, x), min_x)

def compose_transform(translation1, quaternion1, translation2, quaternion2):

    def quaternion_multiply(q1, q2):
        qx1, qy1, qz1, qw1 = q1
        qx2, qy2, qz2, qw2 = q2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        return [qx, qy, qz, qw]
    
    def vector_rotate_by_quaternion(v, q):
        q_conj = [-q[0], -q[1], -q[2], q[3]]
        v_as_quat = [v[0], v[1], v[2], 0]
        rotated_v = quaternion_multiply(quaternion_multiply(q, v_as_quat), q_conj)
        return rotated_v[0:3]

    t1_rotated = vector_rotate_by_quaternion(translation1, quaternion2)
    composed_translation = [t1_rotated[0] + translation2[0], t1_rotated[1] + translation2[1], t1_rotated[2] + translation2[2]]
    composed_rotation = quaternion_multiply(quaternion2, quaternion1)

    return composed_translation, composed_rotation

def cross(vector1, vector2):
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]

def distance(point1, point2):
    return sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def dot(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]

def norm(vector):
    return sqrt(vector[0]**2 + vector[1]**2)

def relative(point1, point2):
    dist = distance(point1[0:2], point2[0:2])
    theta = atan2(point2[1] - point1[1], point2[0] - point1[0])
    
    return dist, theta

def vector_between_theta(vector, left_theta, right_theta):
    left_vector = [cos(left_theta), sin(left_theta)]
    right_vector = [cos(right_theta), sin(right_theta)]
    if cross(vector, left_vector) >= 0 and cross(vector, right_vector) <= 0:
        return True
    else:
        return False

def welzl(points, boundary):
    if len(points) == 0 or len(boundary) == 3:
        if len(boundary) == 0:
            return 0, 0, 0
        elif len(boundary) == 1:
            return boundary[0][0], boundary[0][1], 0
        elif len(boundary) == 2:
            center_x = (boundary[0][0] + boundary[1][0]) / 2
            center_y = (boundary[0][1] + boundary[1][1]) / 2
            radius = distance(boundary[0], boundary[1]) / 2
            return center_x, center_y, radius
        else:
            A = 2 * (boundary[1][0] - boundary[0][0])
            B = 2 * (boundary[1][1] - boundary[0][1])
            C = 2 * (boundary[2][0] - boundary[0][0])
            D = 2 * (boundary[2][1] - boundary[0][1])
            E = boundary[1][0]**2 + boundary[1][1]**2 - boundary[0][0]**2 - boundary[0][1]**2
            F = boundary[2][0]**2 + boundary[2][1]**2 - boundary[0][0]**2 - boundary[0][1]**2
            if A*D - B*C < 1e-6:
                # the three points are collinear, return the circle with the longest distance between two points as diameter
                max_dist = 0
                for i in range(3):
                    for j in range(i+1, 3):
                        dist = distance(boundary[i], boundary[j])
                        if dist > max_dist:
                            max_dist = dist
                            center_x = (boundary[i][0] + boundary[j][0]) / 2
                            center_y = (boundary[i][1] + boundary[j][1]) / 2
                radius = max_dist / 2
                return center_x, center_y, radius
            center_x = (E*D - B*F) / (A*D - B*C)
            center_y = (A*F - E*C) / (A*D - B*C)
            radius = distance([center_x, center_y], boundary[0])
            return center_x, center_y, radius
        
    idx = np.random.randint(len(points))
    point = points[idx]
    points.pop(idx)
    center_x, center_y, radius = welzl(points.copy(), boundary.copy())
    if distance(point, [center_x, center_y]) <= radius:
        return center_x, center_y, radius
    else:
        boundary.append(point)
        return welzl(points.copy(), boundary.copy())

def wraptopi(radian):
    # wrap the angle to [-pi, pi]
    radian = radian % (2*pi)
    
    if radian > pi:
        radian = radian - 2 * pi
    elif radian < -pi:
        radian = radian + 2 * pi

    return radian

def yaw_from_quaternion(quaternion):
    # quaternion = [x, y, z, w]
    siny_cosp = 2 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1])
    cosy_cosp = 1 - 2 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])
    yaw = atan2(siny_cosp, cosy_cosp)
    return yaw

def point_in_circle(point, circle):
    if distance(point, circle[0:2]) <= circle[2]:
        return True
    else:
        return False

def point_in_polygon(point, polygon_vertex_list):
    if len(polygon_vertex_list) < 2:
        return False
    if len(polygon_vertex_list) == 2:
        x, y = point
        x1, y1 = polygon_vertex_list[0]
        x2, y2 = polygon_vertex_list[1]
        # check if point is on the segment
        if cross(np.array([x2 - x1, y2 - y1]), np.array([x - x1, y - y1])) == 0 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        else:
            return False
    
    x, y = point
    inside = False
    for i in range(len(polygon_vertex_list)):
        xi, yi = polygon_vertex_list[i][0], polygon_vertex_list[i][1]
        xj, yj = polygon_vertex_list[i - 1][0], polygon_vertex_list[i - 1][1]

        # check if point is on the edge
        if cross(np.array([xj - xi, yj - yi]), np.array([x - xi, y - yi])) == 0 and min(xi, xj) <= x <= max(xi, xj) and min(yi, yj) <= y <= max(yi, yj):
            return True

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi):
            inside = not inside
    return inside

def circle_intersect_circle(circle1, circle2):
    dist = distance(circle1[0:2], circle2[0:2])
    if dist > circle1[2] + circle2[2] or dist < abs(circle1[2] - circle2[2]):
        return None
    alpha = atan2(circle2[1] - circle1[1], circle2[0] - circle1[0])
    theta = acos((circle1[2]**2 + dist**2 - circle2[2]**2) / (2 * circle1[2] * dist))   # Cosine Theorem
    point1_x = circle1[0] + circle1[2] * cos(alpha + theta)
    point1_y = circle1[1] + circle1[2] * sin(alpha + theta)
    point2_x = circle1[0] + circle1[2] * cos(alpha - theta)
    point2_y = circle1[1] + circle1[2] * sin(alpha - theta)
    return [point1_x, point1_y], [point2_x, point2_y]

def line_in_circle(circle, line_segment):
    # Reference: https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm

    sp = line_segment[0:2]
    ep = line_segment[2:4]

    d = ep - sp
    f = sp - circle[0:2]

    # (f + t*d)^2 - r^2 = d^2 * t^2 + 2*f*d*t + f^2 - r^2 = 0
    a = dot(d, d)
    b = 2*dot(f, d)
    c = dot(f, f) - circle[2]**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0 or a < 1e-6:
        return None
    else:
        t1 = (-b - sqrt(discriminant)) / (2 * a)
        t2 = (-b + sqrt(discriminant)) / (2 * a)

        if t1>=0 and t1<=1 and t2>=0 and t2<=1:
            segment_point1 = sp + t1 * d
            segment_point2 = sp + t2 * d

        elif t1>=0 and t1<=1 and t2 > 1:
            segment_point1 = sp + t1 * d
            segment_point2 = ep
        
        elif t1<0 and t2>=0 and t2<=1:
            segment_point1 = sp
            segment_point2 = sp + t2 * d
        
        elif t1<0 and t2>1:
            segment_point1 = sp
            segment_point2 = ep
        else:
            return None
    
    if distance(segment_point1, segment_point2) < 1e-6:
        return None

    return [segment_point1, segment_point2]

def polygon_in_circle(circle, polygon_edge_list):
    new_vertex_list = []
    for edge in polygon_edge_list:
        segment = line_in_circle(circle, edge)
        if segment is not None:
            vertex1 = list(segment[0])
            vertex2 = list(segment[1])
            if vertex1 not in new_vertex_list:
                new_vertex_list.append(vertex1)
            if vertex2 not in new_vertex_list:
                new_vertex_list.append(vertex2)
    new_vertex_list = list(np.array(vertex) for vertex in new_vertex_list)
    return new_vertex_list

def point2circle(point, circle):
    if point_in_circle(point, circle):
        return 0
    
    dist = distance(point, circle[0:2]) - circle[2]
    return dist

def point2line(point, line_vertex_list):
    sp = line_vertex_list[0]
    ep = line_vertex_list[1]

    l2 = dot(ep - sp, ep - sp)

    if l2 < 1e-6:
        return distance(point, sp)

    t = max(0, min(1, dot(point-sp, ep-sp) / l2))

    closest_point = sp + t * (ep-sp)

    dist = distance(point, closest_point)

    return dist

def point2polygon(point, polygon_vertex_list):
    if point_in_polygon(point, polygon_vertex_list):
        return 0
    
    min_dist = inf
    for i in range(len(polygon_vertex_list)):
        dist = point2line(point, [polygon_vertex_list[i-1], polygon_vertex_list[i]])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def circle2circle(circle1, circle2):
    dist = distance(circle1[0:2], circle2[0:2]) - circle1[2] - circle2[2]
    return max(dist, 0)

def circle2line(circle, line_vertex_list):
    dist = point2line(circle[0:2], line_vertex_list) - circle[2]
    return max(dist, 0)

def circle2polygon(circle, polygon_vertex_list):
    dist = point2polygon(circle[0:2], polygon_vertex_list) - circle[2]
    return dist

def polygon2circle(polygon_vertex_list, circle):
    return circle2polygon(circle, polygon_vertex_list)

def polygon2line(polygon_vertex_list, line_vertex_list):
    sp1 = line_vertex_list[0]
    ep1 = line_vertex_list[1]

    if point_in_polygon(sp1, polygon_vertex_list) or point_in_polygon(ep1, polygon_vertex_list):
        return 0
    
    for i in range(len(polygon_vertex_list)):
        sp2 = polygon_vertex_list[i-1]
        ep2 = polygon_vertex_list[i]
        if cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) < 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) < 0:
            return 0
        elif cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) < 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) == 0:
            return 0
        elif cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) == 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) < 0:
            return 0
        elif cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) == 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) == 0:
            # check if the line segment is overlapped by the polygon edges
            if min(sp1[0], ep1[0]) <= max(sp2[0], ep2[0]) and max(sp1[0], ep1[0]) >= min(sp2[0], ep2[0]) and min(sp1[1], ep1[1]) <= max(sp2[1], ep2[1]) and max(sp1[1], ep1[1]) >= min(sp2[1], ep2[1]):
                return 0

    min_dist = inf
    for i in range(len(polygon_vertex_list)):
        dist = point2line(polygon_vertex_list[i], line_vertex_list)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def polygon2polygon(polygon1_vertex_list, polygon2_vertex_list):
    min_dist = inf
    for i in range(len(polygon1_vertex_list)):
        dist = polygon2line(polygon2_vertex_list, [polygon1_vertex_list[i-1], polygon1_vertex_list[i]])
        if dist < min_dist:
            min_dist = dist
    for i in range(len(polygon2_vertex_list)):
        dist = polygon2line(polygon1_vertex_list, [polygon2_vertex_list[i-1], polygon2_vertex_list[i]])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def circle2circle_collision_time(circle1, circle2, rvx, rvy):
    if circle2circle(circle1, circle2) <= 0:
        return 0
    
    # (x2 - x1 - rvx*t)^2 + (y2 - y1 - rvy*t)^2 < (r1 + r2)^2 ==> (rvx^2 + rvy^2)*t^2 - [2*(x2-x1)*rvx + 2*(y2-y1)*rvy]*t + (x2-x1)^2+(y2-y1)^2-(r1+r2)^2 = 0
    x1, y1, r1 = circle1[0:3]
    x2, y2, r2 = circle2[0:3]

    a = rvx ** 2 + rvy ** 2
    b = -(2 * (x2-x1) * rvx + 2 * (y2-y1) * rvy)
    c = (x2-x1) ** 2 + (y2-y1) ** 2 - (r1+r2) ** 2

    discriminant = b**2 - 4*a*c

    if discriminant < 0 or a < 1e-6:
        t = inf
    else:
        t1 = (-b + sqrt(discriminant)) / (2 * a)
        t2 = (-b - sqrt(discriminant)) / (2 * a)

        t3 = t1 if t1 >= 0 else inf
        t4 = t2 if t2 >= 0 else inf
    
        t = min(t3, t4)

    return t

def circle2line_collision_time(circle, line_vertex_list, rvx, rvy):
    if circle2line(circle, line_vertex_list) <= 0:
        return 0

    circle1 = np.array([line_vertex_list[0][0], line_vertex_list[0][1], 0])
    circle2 = np.array([line_vertex_list[1][0], line_vertex_list[1][1], 0])
    
    t1 = circle2circle_collision_time(circle, circle1, rvx, rvy)
    t2 = circle2circle_collision_time(circle, circle2, rvx, rvy)

    sp = line_vertex_list[0]
    ep = line_vertex_list[1]
    l2 = dot(ep - sp, ep - sp)
    if l2 < 1e-6:
        return min(t1, t2)
    ratio = dot(circle[0:2] - sp, ep - sp) / l2
    projection_point = sp + ratio * (ep - sp)
    projection_dist = distance(circle[0:2], projection_point)
    theta1 = atan2(projection_point[1] - circle[1], projection_point[0] - circle[0])
    theta2 = atan2(rvy, rvx)
    diff_theta = wraptopi(theta2 - theta1)
    speed = norm([rvx, rvy])
    if speed < 1e-6 or cos(diff_theta) < 1e-6:
        t3 = inf
    else:
        t3 = (projection_dist - circle[2]) / (speed * cos(diff_theta))
        if point2line((circle[0] + rvx*t3, circle[1] + rvy*t3), line_vertex_list) > circle[2]: # the collision point is not on the line segment
            t3 = inf

    return min([t1, t2, t3])

def circle2polygon_collision_time(circle, polygon_vertex_list, rvx, rvy):
    if circle2polygon(circle, polygon_vertex_list) <= 0:
        return 0

    min_t = inf
    for i in range(len(polygon_vertex_list)):
        t = circle2line_collision_time(circle, [polygon_vertex_list[i-1], polygon_vertex_list[i]], rvx, rvy)
        if t < min_t:
            min_t = t

    return min_t

def polygon2circle_collision_time(polygon_vertex_list, circle, rvx, rvy):
    return circle2polygon_collision_time(circle, polygon_vertex_list, -rvx, -rvy)

def polygon2line_collision_time(polygon_vertex_list, line_vertex_list, rvx, rvy):
    if polygon2line(polygon_vertex_list, line_vertex_list) <= 0:
        return 0
    
    min_t = inf
    for i in range(len(polygon_vertex_list)):
        temp_circle = np.array([polygon_vertex_list[i][0], polygon_vertex_list[i][1], 0])
        t = circle2line_collision_time(temp_circle, line_vertex_list, rvx, rvy)
        if t < min_t:
            min_t = t
    for i in range(2):
        temp_circle = np.array([line_vertex_list[i][0], line_vertex_list[i][1], 0])
        t = circle2polygon_collision_time(temp_circle, polygon_vertex_list, -rvx, -rvy)
        if t < min_t:
            min_t = t
    return min_t

def polygon2polygon_collision_time(polygon1_vertex_list, polygon2_vertex_list, rvx, rvy):
    if polygon2polygon(polygon1_vertex_list, polygon2_vertex_list) <= 0:
        return 0
    
    min_t = inf
    for i in range(len(polygon1_vertex_list)):
        temp_circle = np.array([polygon1_vertex_list[i][0], polygon1_vertex_list[i][1], 0])
        t = circle2polygon_collision_time(temp_circle, polygon2_vertex_list, rvx, rvy)
        if t < min_t:
            min_t = t
    for i in range(len(polygon2_vertex_list)):
        temp_circle = np.array([polygon2_vertex_list[i][0], polygon2_vertex_list[i][1], 0])
        t = circle2polygon_collision_time(temp_circle, polygon1_vertex_list, -rvx, -rvy)
        if t < min_t:
            min_t = t
    return min_t