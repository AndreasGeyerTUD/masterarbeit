import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


def cantor_pairing(x, y):
    if x is None: return None
    return int((((x + y) * (x + y + 1)) / 2) + y)


def generate_small_hyperspheres_hypercubes(m_rel: int, q: int, max_r: float, min_r: float, hyperspheres: bool,
                                           random_state: int = None):
    hs = []
    for i in range(q):
        c_i = [0] * m_rel
        random.seed(cantor_pairing(random_state, i))
        r = (random.random() * (max_r - min_r)) + min_r
        max_c = 1 - r
        min_c = - max_c

        l = 0
        while True:
            np.random.seed(cantor_pairing(random_state, l))
            for j in np.random.permutation(m_rel):
                k = 0
                while True:
                    random.seed(cantor_pairing(random_state, k + j))
                    c = (random.random() * (max_c - min_c)) + min_c
                    if hyperspheres:
                        upper_bound = math.sqrt(((1 - r) ** 2) - np.sum(np.square(c_i)))
                        if c <= (1 - r) and abs(c) <= upper_bound: break
                    else:
                        if c <= (1 - r): break
                    k += 1
                    if k > 100000: raise TimeoutError("After 100000 executions the stopping condition wasn't met!")

                c_i[j] = c

            if hyperspheres:
                if np.sum(np.square(c_i)) <= (1 - r) ** 2:
                    hs.append((r, c_i))
                    break
            else:
                hs.append((r, c_i))
                break

            l += 1
            if l > 100000: raise TimeoutError("After 100000 executions the stopping condition wasn't met!")

    return hs


def generate_points_inside_hyperspheres(m_rel: int, n: int, c: list, r: float, hyperspheres: bool,
                                        random_state: int = None) -> list:
    xs = []

    for i in range(n):
        x_i = [0] * m_rel

        np.random.seed(cantor_pairing(random_state, i))
        for j in np.random.permutation(m_rel):
            max_x = c[j] + r
            min_x = c[j] - r
            k = 0
            while True:
                random.seed(cantor_pairing(random_state, i + j))
                x = (random.random() * (max_x - min_x)) + min_x
                if hyperspheres:
                    if abs(x - c[j]) > r: continue
                    bound = math.sqrt(r ** 2 - sum([(x_h - c_h) ** 2 if x_h != 0 else 0 for x_h, c_h in zip(x_i, c)]))
                    if c[j] - bound <= x <= c[j] + bound:
                        break
                else:
                    if abs(x - c[j]) <= r: break
                k += 1
                if k > 100000: raise TimeoutError("After 100000 executions the stopping condition wasn't met!")

            x_i[j] = x

        xs.append(x_i)

    return xs


def assign_labels(dataset, hs, q, n):
    labels = np.zeros(shape=(n, q))
    labels = pd.DataFrame(labels, columns=["l{}".format(i) for i in range(q)])
    for point, label in zip(dataset.values, labels.values):
        for idx, (r, c) in enumerate(hs):
            if all(math.sqrt(res ** 2) <= r for res in (np.array(point) - np.array(c))): label[idx] = 1

    return labels


def add_redundant(dataset, n, random_state: int = None):
    red = dataset.sample(n=n, axis='columns', random_state=random_state)

    for i, column in enumerate(red):
        random.seed(cantor_pairing(random_state, i))
        rand = random.random()
        dataset["red{}".format(i)] = red[column] * (rand if rand > 0 else 1)

    return dataset


def add_irrelevant(dataset, n, random_state: int = None):
    for i in range(n):
        cp = cantor_pairing(random_state, i)
        np.random.seed(cp)
        rand_list = np.random.rand(len(dataset))
        random.seed(cp)
        rand = random.random()
        dataset["irr{}".format(i)] = (rand_list - rand)

    return dataset


def add_noise(labels, noise_levels, q: int):
    noisy_labels = []

    for noise in noise_levels:
        new_labels = []
        for label in labels.values:
            new_values = []
            for val in label:
                rand = random.random()
                if rand < noise:
                    val = (val + 1) % 2
                new_values.append(val)
            new_labels.append(new_values)
        noisy_labels.append(pd.DataFrame(new_labels, columns=["l{}".format(i) for i in range(q)]))

    return noisy_labels


def generate(shape: str, m_rel: int, m_irr: int, m_red: int, q: int, n: int, max_r: float = None, min_r: float = None,
             noise_levels: [float] = None, name: str = "Dataset test", random_state: int = None,
             points_distribution: str = None, save_dir: str = None):
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
    if noise_levels is not None and any([m < 0 for m in noise_levels]):
        raise ValueError("my (the levels of noise) must be at least 0 for every level!")
    if m_red > m_rel:
        raise ValueError("m_red must not be larger then m_rel!")
    if min_r >= max_r > 0.8:
        raise ValueError("min_r < max_r <= 0.8 is required!")

    hypershapes = generate_small_hyperspheres_hypercubes(m_rel, q, max_r, min_r, shape == "spheres", random_state)

    if points_distribution == "normal":
        ns = [int(n / q)] * q
    else:
        ns = []
        f = n / np.sum(np.array(hypershapes)[:, 0])

        for r, c in hypershapes:
            ns.append(round(r * f))

    i = 0
    while sum(ns) < n:
        ns[i % q] += 1
        i += 1
    i = 0
    while sum(ns) > n:
        ns[i % q] -= 1 if ns[i % q] > 0 else 0
        i += 1

    dataset = []
    for size, (r, c) in zip(ns, hypershapes):
        dataset.append(generate_points_inside_hyperspheres(m_rel, size, c, r, shape == "spheres", random_state))

    dataset = [item for sublist in dataset for item in sublist]

    dataset = pd.DataFrame(dataset)

    labels = assign_labels(dataset, hypershapes, q, n)

    dataset.rename({i: "rel{}".format(i) for i in range(m_rel)}, axis=1, inplace=True)

    dataset = add_redundant(dataset, m_red, random_state)

    dataset = add_irrelevant(dataset, m_irr, random_state)

    np.random.seed(cantor_pairing(random_state, 1))
    dataset = dataset[np.random.permutation(dataset.columns)]

    noisy_labels = []

    if noise_levels is not None and noise_levels:
        noisy_labels = add_noise(labels.copy(), noise_levels, q)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open("{}/{}_{}_dataset.csv".format(save_dir, name.lower(), shape), "w")as file:
            dataset.to_csv(file, index=False)
        with open("{}/{}_{}_labels.csv".format(save_dir, name.lower(), shape), "w")as file:
            labels.to_csv(file, index=False)
        for ind, noise_level in enumerate(noisy_labels):
            with open("{}/{}_{}_n{}_labels.csv".format(save_dir, name.lower(), shape, ind), "w")as file:
                noise_level.to_csv(file, index=False)

    return dataset, labels, noisy_labels


if __name__ == "__main__":
    dataset, labels, noisy_labels = generate("cubes", 4, 3, 2, 3, 100, None, None, [0.1, 0.2, 0.5], "Test", None, None,
                                             "ml_datagen")
    dataset, labels, noisy_labels = generate("spheres", 4, 3, 2, 3, 100, None, None, [], "Test", None, None,
                                             "ml_datagen")