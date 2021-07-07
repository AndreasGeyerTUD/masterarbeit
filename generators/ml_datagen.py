import math
import random
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd


def cantor_pairing(x: int, y: int = 0) -> Union[int, None]:
    """
    Calculating Cantor Pairing function (https://en.wikipedia.org/wiki/Pairing_function) for seeding random states.
    :param x: Should be the random state. If not given, the function returns None.
    :param y: Any other value you want to pair your random state with.
    :return: The pairing of the two given values.
    """

    if x is None: return None
    return int((((x + y) * (x + y + 1)) / 2) + y)


def generate_small_hypershapes(m_rel: int, q: int, max_r: float, min_r: float, hyperspheres: bool,
                               random_state: int = None) -> list[Tuple[float, list[float]]]:
    """
    As this generator is based on hypercubes and hyperspheres one needs to generate small hypercubes/spheres which will
    later contain the points.
    Pay attention, there are 2 "while True" loops with safety-break after a high number of iterations. If these were
    triggered it might be because of bad luck. Try again a few times (maybe with a different random_state).

    :param m_rel: The number of relevant features to create. Is used to determine the dimensionality of the hypershape.
    :param q: The number of possible labels. For each label there is a hypershape created (overlap of shapes leads to
        multilabel.
    :param max_r: The maximal radius for spheres or half-edge for cubes.
    :param min_r: The minimal radius for spheres or half-edge for cubes.
    :param hyperspheres: Whether to create hyperspheres (True) or hypercubes (False).
    :param random_state: The random state to use. For determinism. This is paired (Cantor Pairing) with another value
        for the random seed.
    :return: The list of hypershapes containing a tuple of the radius/half-edge and the shape center (list of m_rel
        values)
    """

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
                    random.seed(cantor_pairing(random_state, cantor_pairing(k, j)))
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


def generate_points_inside_hypershape(m_rel: int, n: int, c: list[float], r: float, hyperspheres: bool,
                                      random_state: int = None) -> list[list[float]]:
    """
    Populating the beforehand created hypershapes (cubes and spheres) with points (evenly distributed). This function
    populates one given hypershape. So, you have to execute it for every hypershape.

    :param m_rel: The number of relevant features to create. Is used to determine the dimensionality of the points.
    :param n: The number of points to create for this hypershape.
    :param c: The center of the hypershape to populate.
    :param r: The radius/half-edge of the hypershape to populate.
    :param hyperspheres: Whether to populate hyperspheres (True) or hypercubes (False).
    :param random_state: The random state to use. For determinism. This is paired (Cantor Pairing) with another value
        for the random seed.
    :return: A list of m_rel dimensional points for one hypershape. Every point is a list of m_rel values between -1
        and 1 (and inside of the hypershape).
    """

    xs = []

    for i in range(n):
        x_i = [0] * m_rel

        np.random.seed(cantor_pairing(random_state, i))
        for j in np.random.permutation(m_rel):
            max_x = c[j] + r
            min_x = c[j] - r
            k = 0
            while True:
                random.seed(cantor_pairing(random_state, cantor_pairing(i, j)))
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


def assign_labels(dataset: pd.DataFrame, hs: list[Tuple[float, list[float]]], q: int) -> pd.DataFrame:
    """
    Assign the labels for every point. Based on the possible overlap of the hypershapes it is possible that one point
    has several labels (multilabel). This is a list with q entries encoded with One-Hot-Encoding for the labels for
    every point. The membership of a point for a hypershape is calculated by the distance of the point to every shape.

    :param dataset: The dataset (containing only the relevant features) as pandas.DataFrame.
    :param hs: The hypershapes a point might belong to.
    :param q: The number of possible labels.
    :return: A pandas.DataFrame with a One-Hot-Encoded list of labels for every point.
    """

    labels = np.zeros(shape=(len(dataset), q))
    labels = pd.DataFrame(labels, columns=["l{}".format(i) for i in range(q)])
    for point, label in zip(dataset.values, labels.values):
        for idx, (r, c) in enumerate(hs):
            if all(math.sqrt(res ** 2) <= r for res in (np.array(point) - np.array(c))): label[idx] = 1

    return labels


def add_redundant(dataset: pd.DataFrame, m_red: int, random_state: int = None) -> pd.DataFrame:
    """
    Adding redundant features. This means an already existing relevant feature column is copied and multiplied by a
    random factor. So, the new redundant feature is linearly dependent on the relevant feature but doesn't have the
    exact same values.

    :param dataset: The dataset (containing only the relevant features) as pandas.DataFrame.
    :param m_red: The number of redundant features to create.
    :param random_state: The random state to use. For determinism. This is paired (Cantor Pairing) with another value
        for the random seed.
    :return: The pandas.DataFrame dataset extended with the newly created redundant features.
    """

    red = dataset.sample(n=m_red, axis='columns', random_state=random_state)

    for i, column in enumerate(red):
        random.seed(cantor_pairing(random_state, i))
        rand = random.random()
        dataset["red{}".format(i)] = red[column] * (rand if rand > 0 else 1)

    return dataset


def add_irrelevant(dataset: pd.DataFrame, m_irr: int, random_state: int = None) -> pd.DataFrame:
    """
    Adding irrelevant features. These are columns filled with random values.

    :param dataset: The dataset (containing only the relevant and redundant features) as pandas.DataFrame.
    :param m_irr: The number of irrelevant features to create.
    :param random_state: The random state to use. For determinism. This is paired (Cantor Pairing) with another value
        for the random seed.
    :return: The pandas.DataFrame dataset extended with the newly created irrelevant features.
    """

    for i in range(m_irr):
        cp = cantor_pairing(random_state, i)
        np.random.seed(cp)
        rand_list = np.random.rand(len(dataset))
        random.seed(cp)
        rand = random.random()
        dataset["irr{}".format(i)] = (rand_list - rand)

    return dataset


def add_noise(labels: pd.DataFrame, noise_levels: Union[list[float], None], q: int, random_state: int = None) \
        -> list[pd.DataFrame]:
    """
    Create noisy labels. The defined noise_levels tell with which probability a point changes a label.

    Example: noise_levels = [0.1] means, that every entry in the One-Hot-Encoded label list switches with a probability
    of 0.1.

    :param labels: The noiseless labels. These are used as ground truth to add noise.
    :param noise_levels: A list with the probability that the labels switch. 3 entries for example cause 3 different
        results.
    :param q: The number of possible labels. This is used for naming the columns of the new noisy labels DataFrame.
    :param random_state: The random state to use. For determinism. This is paired (Cantor Pairing) with another value
        for the random seed.
    :return: A list containing as many pandas.DataFrame (labels) as there are entries in noise_levels.
    """

    noisy_labels = []

    if noise_levels is not None and noise_levels:
        for i, noise in enumerate(noise_levels):
            new_labels = []
            for j, label in enumerate(labels.values):
                new_values = []
                for k, val in enumerate(label):
                    random.seed(cantor_pairing(random_state, cantor_pairing(k, cantor_pairing(i, j))))
                    rand = random.random()
                    if rand < noise:
                        val = (val + 1) % 2
                    new_values.append(val)
                new_labels.append(new_values)
            noisy_labels.append(pd.DataFrame(new_labels, columns=["l{}".format(i) for i in range(q)]))

    return noisy_labels


def generate(shape: str, m_rel: int, m_irr: int, m_red: int, q: int, n: int, max_r: float = None, min_r: float = None,
             noise_levels: [float] = None, name: str = "Dataset test", random_state: int = None,
             points_distribution: str = None, save_dir: str = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """
    The coordination function for generating the synthetic dataset with the given parameters.

    The dataset and the labels are saved into different csv files located at {save_dir}/{name}_{shape}_*.csv

    :param shape: You can choose between "cubes" and "spheres" which determines whether hypercubes or hyperspheres are
        used.
    :param m_rel: The number of relevant features the dataset should contain.
    :param m_irr: The number of irrelevant features the dataset should contain.
    :param m_red: The number of redundant features the dataset should contain.
    :param q: The number of possible labels.
    :param n: The number of points the dataset should contain.
    :param max_r: The maximal radius for spheres or half-edge for cubes. If not given, the default value of 0.8 is used.
    :param min_r: The minimal radius for spheres or half-edge for cubes. If not given, the default value is calculated
        by round(((q / 10) + 1) / q, 1).
    :param noise_levels: A list with the probability that the labels switch. 3 entries for example cause 3 different
        results.
    :param name: The name of the dataset. Used for the save_file.
    :param random_state: The random state to use. For determinism. This is paired (Cantor Pairing) with another value
        for the random seed.
    :param points_distribution: How to distribute the number of points on the different hypershapes. Default is a
        weighted distribution based on the size of the hypershape. Other values are "uniform" for a uniform distribution.
    :param save_dir: The directory where the dataset and the labels should be saved.
    :return: A tuple containing the created dataset, the corresponding labels (without noise) and a list containing the
        noisy labels (one pandas.DataFrame of labels for every noise level).
    """

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
    if noise_levels is not None and any([m < 0 or m > 1 for m in noise_levels]):
        raise ValueError(
            "noise_levels must be at least 0 (no value changes) and at most 1 (every value changes) for every level!")
    if m_red > m_rel:
        raise ValueError("m_red must not be larger then m_rel!")
    if min_r >= max_r > 0.8:
        raise ValueError("min_r < max_r <= 0.8 is required!")

    hypershapes = generate_small_hypershapes(m_rel, q, max_r, min_r, shape == "spheres", random_state)

    if points_distribution == "uniform":
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
        dataset.append(generate_points_inside_hypershape(m_rel, size, c, r, shape == "spheres", random_state))

    dataset = [item for sublist in dataset for item in sublist]

    dataset = pd.DataFrame(dataset)

    labels = assign_labels(dataset, hypershapes, q)

    dataset.rename({i: "rel{}".format(i) for i in range(m_rel)}, axis=1, inplace=True)

    dataset = add_redundant(dataset, m_red, random_state)

    dataset = add_irrelevant(dataset, m_irr, random_state)

    np.random.seed(cantor_pairing(random_state, 1))
    dataset = dataset[np.random.permutation(dataset.columns)]

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
    # Todo: check what's the reason for not working with random_state set
    dataset, labels, noisy_labels = generate("cubes", 4, 3, 2, 3, 100, None, None, [0.1, 0.2, 0.5], "Test", None, None,
                                             "ml_datagen")
    dataset, labels, noisy_labels = generate("spheres", 4, 3, 2, 3, 100, None, None, [], "Test", None, None,
                                             "ml_datagen")
