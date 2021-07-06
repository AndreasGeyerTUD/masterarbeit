import math
import random
from numpy.random import permutation
import numpy as np


def generate_small_hyperspheres_hypercubes(m_rel: int, q: int, max_r: float, min_r: float, hyperspheres: bool):
    hs = []
    for i in range(q):
        c_i = [0] * m_rel
        r = (random.random() * (max_r - min_r)) + min_r
        max_c = 1 - r
        min_c = - max_c

        i = 0
        while True:
            for j in permutation(m_rel):
                k = 0
                while True:
                    c = (random.random() * (max_c - min_c)) + min_c
                    upper_bound = math.sqrt(((1 - r) ** 2) - np.sum(np.square(c_i)))
                    if c <= (1 - r) and -upper_bound <= c <= upper_bound: break
                    k += 1
                    if k > 10000: raise TimeoutError("After 10000 executions the stopping condition wasn't met!")

                # if hyperspheres:
                # max_c = update_min_max_c(r, c_i)
                # max_c = math.sqrt(((1 - r) ** 2) - np.sum(np.square(c_i)))
                # min_c = - max_c

                c_i[j] = c

            if np.sum(np.square(c_i)) <= (1 - r) ** 2:
                hs.append((r, c_i))
                break

            i += 1
            if i > 10000: raise TimeoutError("After 10000 executions the stopping condition wasn't met!")

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


def hyperspheres(m_rel: int, m_irr: int, m_red: int, q: int, n: int, max_r: float = None, min_r: float = None,
                 my: [float] = [0.05, 0.1], name: str = "Dataset test"):
    if max_r is None:
        max_r = 0.8
    if min_r is None:
        min_r = round(((q / 10) + 1) / q, 1)

    if m_rel <= 0:
        raise ValueError("m_rel (the number of relevant features) must be larger than 0!")
    if m_irr < 0:
        raise ValueError("m_irr (the number of irrelevant features) must be at least 0!")
    if m_red < 0:
        raise ValueError("m_red (the number of redundant features) must be at least 0!")
    if q <= 0:
        raise ValueError("q (the number of labels) must be larger than 0!")
    if n <= 0:
        raise ValueError("n (the number of samples) must be larger than 0!")
    if max_r <= 0:
        raise ValueError("max_r (the maximum radius) must be larger than 0!")
    if min_r <= 0:
        raise ValueError("min_r (the minimum radius) must be larger than 0!")
    if my is not None and any(m < 0 for m in my):
        raise ValueError("my (the levels of noise) must be at least 0 for every level!")
    if m_red > m_rel:
        raise ValueError("m_red must not be larger then m_rel!")
    if min_r >= max_r > 0.8:
        raise ValueError("min_r < max_r <= 0.8 is required!")

    hyperspheres = generate_small_hyperspheres_hypercubes(m_rel, q, max_r, min_r, True)

    print()


if __name__ == "__main__":
    hyperspheres(4, 0, 0, 3, 100, None, None, [], "Test")
