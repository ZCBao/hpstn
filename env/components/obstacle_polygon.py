import numpy as np
from env.utils.utils import welzl

class ObstaclePolygon:
    def __init__(self, vertex_list):
        self.vertex_list = list(np.array(vertex) for vertex in vertex_list)
        assert len(self.vertex_list) >= 3
        self.generate_edges()

    def generate_edges(self):
        self.edge_list = []

        for i in range(len(self.vertex_list)):
            edge = np.concatenate([self.vertex_list[i], self.vertex_list[(i + 1) % len(self.vertex_list)]])
            self.edge_list.append(edge)
    
    def obs_state(self):
        # px, py, vx, vy, radius
        cx, cy, r = self.min_enclosing_circle()
        return np.array([cx, cy, 0, 0, r])
    
    def min_enclosing_circle(self):
        return welzl(self.vertex_list.copy(), [])