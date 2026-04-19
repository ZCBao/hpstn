import numpy as np

class ObstacleLine:
    def __init__(self, vertex_list):
        self.vertex_list = [np.array(vertex_list[0]), np.array(vertex_list[1])]
        self.segment = np.concatenate([vertex_list[0], vertex_list[1]])