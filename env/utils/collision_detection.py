import numpy as np
from env.utils.utils import cross, distance, dot, point_in_circle, point_in_polygon

# world: origin, width, height
# matrix: np.array([width, height])
# circle: np.array([x, y, r])
# line_segment: np.array([x1, y1, x2, y2])
# polygon: edge_list, vertex_list

def collision_circle_world(circle, origin=np.zeros(2), width=10.0, height=10.0):
    circle_min_x = circle[0] - circle[2]
    circle_max_x = circle[0] + circle[2]
    circle_min_y = circle[1] - circle[2]
    circle_max_y = circle[1] + circle[2]

    world_min_x = origin[0]
    world_max_x = origin[0] + width
    world_min_y = origin[1]
    world_max_y = origin[1] + height

    if (circle_min_x < world_min_x) or (circle_max_x > world_max_x) or (circle_min_y < world_min_y) or (circle_max_y > world_max_y):
        return True
    else:
        return False

def collision_circle_matrix(circle, matrix, origin=np.zeros(2), resolution=0.1):
    circle_min_x = circle[0] - circle[2]
    circle_max_x = circle[0] + circle[2]
    circle_min_y = circle[1] - circle[2]
    circle_max_y = circle[1] + circle[2]

    matrix_min_x = origin[0]
    matrix_max_x = origin[0] + matrix.shape[0] * resolution
    matrix_min_y = origin[1]
    matrix_max_y = origin[1] + matrix.shape[1] * resolution

    if (circle_min_x > matrix_max_x) or (circle_max_x < matrix_min_x) or (circle_min_y > matrix_max_y) or (circle_max_y < matrix_min_y):
        return False

    circle_min_i = int(np.floor((circle_min_x - origin[0]) / resolution))
    circle_max_i = int(np.ceil((circle_max_x - origin[0]) / resolution))
    circle_min_j = int(np.floor((circle_min_y - origin[1]) / resolution))
    circle_max_j = int(np.ceil((circle_max_y - origin[1]) / resolution))

    for i in range(circle_min_i, circle_max_i+1):
        for j in range(circle_min_j, circle_max_j+1):
            if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]:
                if matrix[i, j]:
                    grid_min_x = origin[0] + i * resolution
                    grid_max_x = origin[0] + (i + 1) * resolution
                    grid_min_y = origin[1] + j * resolution
                    grid_max_y = origin[1] + (j + 1) * resolution
                    if point_in_circle(np.array([grid_min_x, grid_min_y]), circle):
                        return True
                    if point_in_circle(np.array([grid_max_x, grid_min_y]), circle):
                        return True
                    if point_in_circle(np.array([grid_max_x, grid_max_y]), circle):
                        return True
                    if point_in_circle(np.array([grid_min_x, grid_max_y]), circle):
                        return True
    return False

def collision_circle_circle(circle1, circle2):
    dist = distance(circle1[0:2], circle2[0:2])

    if dist <= circle1[2] + circle2[2]:
        return True
    else:
        return False

def collision_circle_line(circle, line_segment):
    circle_min_x = circle[0] - circle[2]
    circle_max_x = circle[0] + circle[2]
    circle_min_y = circle[1] - circle[2]
    circle_max_y = circle[1] + circle[2]

    line_min_x = min(line_segment[0], line_segment[2])
    line_max_x = max(line_segment[0], line_segment[2])
    line_min_y = min(line_segment[1], line_segment[3])
    line_max_y = max(line_segment[1], line_segment[3])

    if (circle_min_x > line_max_x) or (circle_max_x < line_min_x) or (circle_min_y > line_max_y) or (circle_max_y < line_min_y):
        return False
    
    center = circle[0:2]
    sp = line_segment[0:2]
    ep = line_segment[2:4]
    
    l2 = dot(ep - sp, ep - sp)

    if l2 == 0.0:
        dist = distance(center, sp)
        if dist <= circle[2]:
            return True
        else:
            return False

    t = max(0, min(1, dot(center-sp, ep-sp) / l2))

    closest_point = sp + t * (ep-sp)

    dist = distance(center, closest_point)
    
    if dist <= circle[2]:
        return True
    else:
        return False

def collision_circle_polygon(circle, polygon):
    circle_min_x = circle[0] - circle[2]
    circle_max_x = circle[0] + circle[2]
    circle_min_y = circle[1] - circle[2]
    circle_max_y = circle[1] + circle[2]

    polygon_min_x = min(vertex[0] for vertex in polygon.vertex_list)
    polygon_max_x = max(vertex[0] for vertex in polygon.vertex_list)
    polygon_min_y = min(vertex[1] for vertex in polygon.vertex_list)
    polygon_max_y = max(vertex[1] for vertex in polygon.vertex_list)

    if (circle_min_x > polygon_max_x) or (circle_max_x < polygon_min_x) or (circle_min_y > polygon_max_y) or (circle_max_y < polygon_min_y):
        return False
    
    # check if the circle center is inside the polygon
    if point_in_polygon(circle[0:2], polygon.vertex_list):
        return True
    
    for edge in polygon.edge_list:
        if collision_circle_line(circle, edge):
            return True
    
    return False

def collision_polygon_world(polygon, origin=np.zeros(2), width=10.0, height=10.0):
    polygon_min_x = min(vertex[0] for vertex in polygon.vertex_list)
    polygon_max_x = max(vertex[0] for vertex in polygon.vertex_list)
    polygon_min_y = min(vertex[1] for vertex in polygon.vertex_list)
    polygon_max_y = max(vertex[1] for vertex in polygon.vertex_list)

    world_min_x = origin[0]
    world_max_x = origin[0] + width
    world_min_y = origin[1]
    world_max_y = origin[1] + height

    if (polygon_min_x < world_min_x) or (polygon_max_x > world_max_x) or (polygon_min_y < world_min_y) or (polygon_max_y > world_max_y):
        return True
    else:
        return False

def collision_polygon_matrix(polygon, matrix, origin=np.zeros(2), resolution=0.1):
    polygon_min_x = min(vertex[0] for vertex in polygon.vertex_list)
    polygon_max_x = max(vertex[0] for vertex in polygon.vertex_list)
    polygon_min_y = min(vertex[1] for vertex in polygon.vertex_list)
    polygon_max_y = max(vertex[1] for vertex in polygon.vertex_list)

    matrix_min_x = origin[0]
    matrix_max_x = origin[0] + matrix.shape[0] * resolution
    matrix_min_y = origin[1]
    matrix_max_y = origin[1] + matrix.shape[1] * resolution

    if (polygon_min_x > matrix_max_x) or (polygon_max_x < matrix_min_x) or (polygon_min_y > matrix_max_y) or (polygon_max_y < matrix_min_y):
        return False

    polygon_min_i = int(np.floor((polygon_min_x - origin[0]) / resolution))
    polygon_max_i = int(np.ceil((polygon_max_x - origin[0]) / resolution))
    polygon_min_j = int(np.floor((polygon_min_y - origin[1]) / resolution))
    polygon_max_j = int(np.ceil((polygon_max_y - origin[1]) / resolution))

    for i in range(polygon_min_i, polygon_max_i+1):
        for j in range(polygon_min_j, polygon_max_j+1):
            if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]:
                if matrix[i, j]:
                    grid_min_x = origin[0] + i * resolution
                    grid_max_x = origin[0] + (i + 1) * resolution
                    grid_min_y = origin[1] + j * resolution
                    grid_max_y = origin[1] + (j + 1) * resolution
                    if point_in_polygon(np.array([grid_min_x, grid_min_y]), polygon.vertex_list):
                        return True
                    if point_in_polygon(np.array([grid_max_x, grid_min_y]), polygon.vertex_list):
                        return True
                    if point_in_polygon(np.array([grid_max_x, grid_max_y]), polygon.vertex_list):
                        return True
                    if point_in_polygon(np.array([grid_min_x, grid_max_y]), polygon.vertex_list):
                        return True
    return False

def collision_polygon_circle(polygon, circle):
    return collision_circle_polygon(circle, polygon)

def collision_polygon_line(polygon, line_segment):
    polygon_min_x = min(vertex[0] for vertex in polygon.vertex_list)
    polygon_max_x = max(vertex[0] for vertex in polygon.vertex_list)
    polygon_min_y = min(vertex[1] for vertex in polygon.vertex_list)
    polygon_max_y = max(vertex[1] for vertex in polygon.vertex_list)

    line_min_x = min(line_segment[0], line_segment[2])
    line_max_x = max(line_segment[0], line_segment[2])
    line_min_y = min(line_segment[1], line_segment[3])
    line_max_y = max(line_segment[1], line_segment[3])

    if (polygon_min_x > line_max_x) or (polygon_max_x < line_min_x) or (polygon_min_y > line_max_y) or (polygon_max_y < line_min_y):
        return False
    
    sp1 = line_segment[0:2]
    ep1 = line_segment[2:4]

    # check if the line segment is inside the polygon
    if point_in_polygon(sp1, polygon.vertex_list) or point_in_polygon(ep1, polygon.vertex_list):
        return True
    
    for edge in polygon.edge_list:
        sp2 = edge[0:2]
        ep2 = edge[2:4]
        if cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) < 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) < 0:
            return True
        elif cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) < 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) == 0:
            return True
        elif cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) == 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) < 0:
            return True
        elif cross(ep1 - sp1, sp2 - sp1) * cross(ep1 - sp1, ep2 - sp1) == 0 and cross(ep2 - sp2, sp1 - sp2) * cross(ep2 - sp2, ep1 - sp2) == 0:
            # check if the line segment is overlapped by the polygon edges
            if min(sp1[0], ep1[0]) <= max(sp2[0], ep2[0]) and max(sp1[0], ep1[0]) >= min(sp2[0], ep2[0]) and min(sp1[1], ep1[1]) <= max(sp2[1], ep2[1]) and max(sp1[1], ep1[1]) >= min(sp2[1], ep2[1]):
                return True
    
    return False

def collision_polygon_polygon(polygon1, polygon2):
    polygon1_min_x = min(vertex[0] for vertex in polygon1.vertex_list)
    polygon1_max_x = max(vertex[0] for vertex in polygon1.vertex_list)
    polygon1_min_y = min(vertex[1] for vertex in polygon1.vertex_list)
    polygon1_max_y = max(vertex[1] for vertex in polygon1.vertex_list)

    polygon2_min_x = min(vertex[0] for vertex in polygon2.vertex_list)
    polygon2_max_x = max(vertex[0] for vertex in polygon2.vertex_list)
    polygon2_min_y = min(vertex[1] for vertex in polygon2.vertex_list)
    polygon2_max_y = max(vertex[1] for vertex in polygon2.vertex_list)

    if (polygon1_min_x > polygon2_max_x) or (polygon1_max_x < polygon2_min_x) or (polygon1_min_y > polygon2_max_y) or (polygon1_max_y < polygon2_min_y):
        return False
    
    for edge in polygon1.edge_list:
        if collision_polygon_line(polygon2, edge):
            return True
    for edge in polygon2.edge_list:
        if collision_polygon_line(polygon1, edge):
            return True
    return False