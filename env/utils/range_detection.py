import numpy as np
from env.utils.utils import cross, distance, dot, norm

# line:   np.array([[x1, y1], [x2, y2]])
# matrix: np.array([width * height])
# circle: np.array([x, y, r])

# def range_line_matrix(line, matrix, origin=np.zeros(2), resolution=0.1):
#     start_point = line[0]
#     diff = line[1] - line[0]
#     line_len = norm(diff)

#     cur_len = 0
#     while cur_len <= line_len:
#         cur_point = start_point + diff * cur_len / line_len
#         index = np.floor((cur_point - origin) / resolution)

#         if index[0] < 0 or index[0] >= matrix.shape[0] or index[1] < 0 or index[1] >= matrix.shape[1]:
#             dist = distance(cur_point, start_point)
#             return True, cur_point, dist
#         elif matrix[int(index[0]), int(index[1])]:
#             dist = distance(cur_point, start_point)
#             return True, cur_point, dist

#         cur_len = cur_len + resolution

#     return False, None, None

def range_line_matrix(line, matrix, origin=np.zeros(2), resolution=0.1):
    # Amanatides-Woo algorithm
    start = line[0]
    end = line[1]
    dir_vec = end - start
    line_len = norm(dir_vec)
    if line_len < 1e-6:
        return False, None, None

    dir_norm = dir_vec / line_len

    grid_origin = origin
    start_grid = (start - grid_origin) / resolution
    ix = int(np.floor(start_grid[0]))
    iy = int(np.floor(start_grid[1]))

    grid_width, grid_height = matrix.shape[0], matrix.shape[1]

    step_x = 1 if dir_norm[0] > 0 else -1 if dir_norm[0] < 0 else 0
    step_y = 1 if dir_norm[1] > 0 else -1 if dir_norm[1] < 0 else 0

    if dir_vec[0] != 0:
        t_max_x = ( (ix + (step_x > 0)) * resolution + grid_origin[0] - start[0] ) / dir_vec[0]
        t_delta_x = (step_x * resolution) / dir_vec[0]
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    if dir_vec[1] != 0:
        t_max_y = ( (iy + (step_y > 0)) * resolution + grid_origin[1] - start[1] ) / dir_vec[1]
        t_delta_y = (step_y * resolution) / dir_vec[1]
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    while True:
        if ix < 0 or ix >= grid_width or iy < 0 or iy >= grid_height:
            t = min(t_max_x, t_max_y)
            if t > 1.0:
                return False, None, None
            intersection = start + t * dir_vec
            dist = distance(intersection, start)
            return True, intersection, dist
        
        if matrix[ix, iy]:
            t = min(t_max_x, t_max_y)
            if t > 1.0:
                return False, None, None
            intersection = start + t * dir_vec
            dist = distance(intersection, start)
            return True, intersection, dist
        
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            ix += step_x
        else:
            t_max_y += t_delta_y
            iy += step_y

        if t_max_x > 1.0 and t_max_y > 1.0:
            return False, None, None

def range_line_circle(line, circle):
    sp = line[0]
    ep = line[1]

    d = ep - sp
    f = sp - circle[0:2]

    # (f + t*d)^2 - r^2 = d^2 * t^2 + 2*f*d*t + f^2 - r^2 = 0
    a = dot(d, d)
    b = 2*dot(f, d)
    c = dot(f, f) - circle[2]**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0 or a < 1e-6:
        return False, None, None
    else:
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)

        if 0 <= t1 <= 1:
            intersection_point = sp + t1 * d
            dist = distance(intersection_point, sp)
            return True, intersection_point, dist
        if 0 <= t2 <= 1:
            intersection_point = sp + t2 * d
            dist = distance(intersection_point, sp)
            return True, intersection_point, dist

        return False, None, None
    
def range_line_line(line1, line2):
    # Reference https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    # p, p+r, q, q+s: p+t*r = q+u*s

    if (max(line1[0][0], line1[1][0]) < min(line2[0][0], line2[1][0]) or max(line1[0][1], line1[1][1]) < min(line2[0][1], line2[1][1]) or
        min(line1[0][0], line1[1][0]) > max(line2[0][0], line2[1][0]) or min(line1[0][1], line1[1][1]) > max(line2[0][1], line2[1][1])):
        return False, None, None

    p = line1[0]
    r = line1[1] - line1[0]
    q = line2[0]
    s = line2[1] - line2[0]

    temp1 = cross(r, s)
    temp2 = cross(q - p, r)

    if temp1 == 0 and temp2 == 0:
        # collinear
        t0 = dot(q - p, r) / dot(r, r) if dot(r, r) != 0 else 0.0
        t1 = t0 + dot(s, r) / dot(r, r) if dot(r, r) != 0 else 0.0
        t_min = min(t0, t1)
        t_max = max(t0, t1)

        if t_max >= 0 and t_min < 0:
            intersection_point = p
            dist = 0
            return True, intersection_point, dist
        elif t_min >=0 and t_min <= 1:
            intersection_point = p + t_min * r
            dist = distance(intersection_point, p)
            return True, intersection_point, dist
        else:
            return False, None, None
    
    elif temp1 == 0 and temp2 != 0:
        # parallel and non-intersecting
        return False, None, None

    elif temp1 != 0:
        t = cross(q - p, s) / temp1
        u = cross(q - p, r) / temp1

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = p + t * r
            dist = distance(intersection_point, p)
            return True, intersection_point, dist
        else: 
            return False, None, None

    else:
        # not parallel and not intersect
        return False, None, None