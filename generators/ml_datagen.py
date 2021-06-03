import math
import random
from numpy.random import permutation
import numpy as np


def generate_small_hyperspheres_hypercubes(m_rel: int, q: int, min_r: float, max_r: float, hyperspheres: bool):
    hs = []
    for i in range(q):
        c_i = [0] * m_rel
        r = (random.random() * (max_r - min_r)) + min_r
        max_c = 1 - r
        min_c = - max_c

        for j in permutation(m_rel):
            c = (random.random() * (max_c - min_c)) + min_c

            if hyperspheres:
                max_c = update_min_max_c(r, c_i)
                min_c = - max_c

            c_i[j] = c

        hs.append((r, c_i))

    return hs


def update_min_max_c(r, c_i):
    return math.sqrt(((1 - r) ** 2) - np.sum(np.square(c_i)))


def generate_points_inside_hyperspheres(m_rel: int, n: int, c: list, r: list, hyperspheres: bool):
    for i in range(n):
        x_i = [0] * m_rel
        max_x = c[i] - r
        min_x = c[i] + r

        for j in permutation(m_rel):
            x = (random.random() * (max_x - min_x)) + min_x

            if hyperspheres:
                new_min_max = update_min_max_x(min_x)
                min_x = c[i][j] - new_min_max
                max_x = c[i][j] + new_min_max

            x_i[j] = x

    return x_i


def update_min_max_x(r, x_i, c_i):
    return


# def generate_small_hypercubes(q: int, minR: float, maxR: float):
#     hc = []
#     for i in range(q):
#         C = {}
#         e = (random.random() * (maxR - minR)) + minR
#         maxC = 1 - e
#         minC = - maxC
#
#         for j in range():
#             c = (random.random() * (maxC - minC)) + minC
#             C.update(c)
#
#         hc.append((e, C))
#
#     return hc


def generate_points_inside_hypercubes(N: int, c, e):
    for i in range(N):
        X = {}
        maxX = c - e
        minX = c + e

        for j in range():
            x = (random.random() * (maxX - minX)) + minX
            X.update(x)

    return X
