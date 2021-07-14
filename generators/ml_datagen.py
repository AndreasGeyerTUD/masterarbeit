import math
import random
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.spatial.distance import euclidean
from scipy.special import gamma, betainc


def calc_hypersphere_iou(hs: list[Tuple[float, list[float]]], new_hs: Tuple[float, list[float]], m_rel: int) \
        -> list[float]:
    """
    As the name suggest this method calculates the n-dimensional Intersection over Union (IoU). This value is as exact
    as it gets. For each element in hs the IoU with the new_hs is calculated.

    :param hs: A list of hyperspheres containing tuple of radius and corresponding center-points.
    :param new_hs: The reference hypersphere. For every element of hs the IoU with this hypersphere is calculated. It
        is a tuple of radius and center-point.
    :param m_rel: The number of relevant features to create. Is used to determine the dimensionality of the hypershape.
    :return: A list of IoUs in the same order as hs. So, for every element of hs there is the IoU with new_hs.
    """

    if hs is None or len(hs) == 0 or not hs:
        return [-1]

    ious = []
    new_r, new_c = new_hs
    new_v = calc_hypersphere_volume(new_r, m_rel)

    for r, c in hs:
        intersection = calc_hypersphere_intersection_volume(c, r, new_c, new_r, m_rel)
        volume = calc_hypersphere_volume(r, m_rel)
        if intersection == -1: intersection = min(volume, new_v)
        union = volume + new_v - intersection

        ious.append(intersection / union)

    return ious


def calc_hypersphere_intersection_volume(center_1: list[float], r_1: float, center_2: list[float], r_2: float,
                                         n: int) -> float:
    """
    This method calculates the intersection of two given Hyperspheres.

    :param center_1: The center-point of the first Hypersphere.
    :param r_1: The radius of the first Hypersphere.
    :param center_2: The center-point of the second Hypersphere.
    :param r_2: The radius of the second Hypersphere.
    :param n: The dimension of the Hyperspheres.
    :return: A value for the intersection of the two given Hyperspheres.
    """

    d = euclidean(center_1, center_2)

    if d >= r_1 + r_2:
        return 0
    elif d <= abs(r_1 - r_2):
        return -1
    else:
        c_1 = ((d ** 2) + (r_1 ** 2) - (r_2 ** 2)) / (2 * d)
        c_2 = ((d ** 2) - (r_1 ** 2) + (r_2 ** 2)) / (2 * d)
        return calc_hypersphere_cap_volume(r_1, c_1, n) + calc_hypersphere_cap_volume(r_2, c_2, n)


def calc_hypersphere_cap_volume(r: float, c: float, n: int):
    """
    Calculation of the volume of the actual intersecting hypershape.

    :param r: Radius of the whole Hypersphere.
    :param c: The c from calc_hypersphere_intersection_volume.
    :param n: The dimension of the Hypersphere.
    :return: Volume of the actual intersecting hypershape.
    """
    term = (math.pi ** (n / 2) * r ** n) / gamma((n / 2) + 1)
    if c < 0:
        return term - calc_hypersphere_cap_volume(r, -c, n)
    else:
        a = (n + 1) / 2
        x = 1 - ((c ** 2) / (r ** 2))
        return 0.5 * term * betainc(a, 0.5, x)


def calc_hypersphere_volume(r: float, n: int) -> float:
    """
    Calculates the volume of a given Hypersphere.

    :param r: The radius of the Hypersphere.
    :param n: The dimension of the Hypersphere.
    :return: The volume of the Hypersphere.
    """
    return (math.pi ** (n / 2) * r ** n) / gamma((n / 2) + 1)


def _print_iou_matrix(hs: list[Tuple[float, list[float]]], m_rel: int, hyperspheres: bool) -> None:
    """
    Helper function for printing the half-matrix of the IoU from each Hypershape with each other.

    :param hs: A list of hyperspheres containing tuples of radius and corresponding center-point.
    :param m_rel: The dimension of the Hypershape.
    :param hyperspheres: Whether the shape is a Hypersphere (True) or a Hypercube (False).
    :return: None
    """
    result = np.zeros((len(hs), len(hs)))
    for i, h_i in enumerate(hs):
        for j, h_j in enumerate(hs):
            if i <= j:
                continue
            result[i][j] = calc_hypersphere_iou([h_i], h_j, m_rel)[0] if hyperspheres else \
                approximate_hypercube_iou([h_i], h_j, m_rel)[0]

    print(result)


def approximate_hypercube_iou(hs: list[Tuple[float, list[float]]], new_hs: Tuple[float, list[float]], m_rel: int) \
        -> list[float]:
    """
    As it is non-trivial to calculate the intersection of two Hypercubes, the Intersection and therefore IoU is only
    an approximation. Tested with several combinations of parameters and the accuracy was sufficient in all cases. To
    use this method is time consuming with large m_rel.

    :param hs: A list of hyperspheres containing tuples of radius and corresponding center-point.
    :param new_hs: The reference Hypercube. For every element of hs the IoU with this Hypercube is calculated. It
        is a tuple of radius and center-point.
    :param m_rel: The dimension of the Hypercube.
    :return: A list of IoUs in the same order as hs. So, for every element of hs there is the IoU with new_hs.
    """
    if hs is None or len(hs) == 0 or not hs:
        return [-1]

    number_rand_points = 10000 * m_rel

    inside_ref = [0] * len(hs)

    new_r, new_c = new_hs
    new_v = calc_hypercube_volume(new_r, m_rel)

    for i, (r, c) in enumerate(hs):
        j = 0
        while j < number_rand_points:
            max_x = np.array(c) + r
            min_x = np.array(c) - r
            p = np.random.rand(m_rel) * (max_x - min_x) + min_x
            p_inside_hc = is_point_inside_hypercube(p, c, r)
            if p_inside_hc:
                j += 1
            if p_inside_hc and is_point_inside_hypercube(p, new_c, new_r):
                inside_ref[i] += 1

    result = []

    for (r, c), k in zip(hs, inside_ref):
        v = calc_hypercube_volume(r, m_rel)
        inter = (k / number_rand_points) * v
        union = v + new_v - inter
        result.append(inter / union)

    return result


def calc_hypercube_volume(r: float, n: int) -> float:
    """
    Calculates the volume of a given Hypercube.

    :param r: The radius of the Hypercube.
    :param n: The dimension of the Hypercube.
    :return: The volume of the Hypercube.
    """
    return (r * 2) ** n


def is_point_inside_hypercube(point: list[float], c: list[float], r: float) -> bool:
    """
    This method checks whether a given point is inside a given Hypercube.

    :param point: The point to check.
    :param c: The center-point of the Hypercube.
    :param r: The half-edge-length of the Hypercube.
    :return: Whether the point is inside or not.
    """
    diff = np.subtract(point, c)
    return np.all(np.absolute(diff) <= r)


def check_iou_threshold(hs: list[Tuple[float, list[float]]], r: float, c_i: list[float], m_rel: int, hyperspheres: bool,
                        iou_threshold: Union[float, list[float]] = None) -> bool:
    """
    Check whether the given Hypershape meets the given restrictions for the IoU. To apply iou-restrictions has a high
    negative impact on the performance of the dataset generation.

    :param hs: A list of hyperspheres containing tuples of radius and corresponding center-point.
    :param r: The radius/half-edge-length for the Hypershape to test.
    :param c_i: The center-point for the Hypershape to test.
    :param m_rel: The dimension of the Hypershape.
    :param hyperspheres: Whether the Hypershape is a Hypersphere (True) or a Hypercube (False).
    :param iou_threshold: The restrictions to apply for the IoU. If None than no restrictions are applied. If a single
        float value, than this is the upper bound of the IoU for every pairing of Hypershapes, so, no Hypershape has an
        IoU larger than this float value with any other Hypershape. If a list of floats, than this list should contain
        exactly 2 values. The first value is the lower bound which enforces every Hypershape to have at least this value
        as IoU with at least one other Hypershape, so, it is possible for two Hypershapes to have no intersection, but
        it ensures that every Hypershape intersects with at least one other Hypershape. The second value is used as
        upper bound and in the same way a single float value would be applied.
    :return: Whether the IoU conditions are met.
    """
    if iou_threshold is None: return True
    ious = np.array(calc_hypersphere_iou(hs, (r, c_i), m_rel)) if hyperspheres else np.array(
        approximate_hypercube_iou(hs, (r, c_i), m_rel))
    print(ious)
    if np.all(ious == -1): return True
    if isinstance(iou_threshold, float):
        return np.all(ious <= iou_threshold)
    elif isinstance(iou_threshold, list):
        return np.any(iou_threshold[0] <= ious) and np.all(ious <= iou_threshold[1])


def generate_small_hypershapes(m_rel: int, q: int, max_r: float, min_r: float, hyperspheres: bool,
                               iou_threshold: Union[float, list[float]] = None) -> list[Tuple[float, list[float]]]:
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
    :param iou_threshold: The restrictions to apply for the IoU. If None than no restrictions are applied. If a single
        float value, than this is the upper bound of the IoU for every pairing of Hypershapes, so, no Hypershape has an
        IoU larger than this float value with any other Hypershape. If a list of floats, than this list should contain
        exactly 2 values. The first value is the lower bound which enforces every Hypershape to have at least this value
        as IoU with at least one other Hypershape, so, it is possible for two Hypershapes to have no intersection, but
        it ensures that every Hypershape intersects with at least one other Hypershape. The second value is used as
        upper bound and in the same way a single float value would be applied.
    :return: The list of hypershapes containing a tuple of the radius/half-edge and the shape center (list of m_rel
        values)
    """

    hs = []
    for i in range(q):
        c_i = [0] * m_rel
        r = (random.random() * (max_r - min_r)) + min_r
        max_c = 1 - r
        min_c = - max_c

        l = 0
        while True:
            for j in np.random.permutation(m_rel):
                k = 0
                while True:
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
                    if check_iou_threshold(hs, r, c_i, m_rel, hyperspheres, iou_threshold):
                        hs.append((r, c_i))
                        break
            else:
                if check_iou_threshold(hs, r, c_i, m_rel, hyperspheres, iou_threshold):
                    hs.append((r, c_i))
                    break

            l += 1
            if l > 100000: raise TimeoutError("After 100000 executions the stopping condition wasn't met!")

    # _print_iou_matrix(hs, m_rel, hyperspheres)

    return hs


def move_point(point: list[float], mov_vector: list[float]):
    return (np.array(point) + (
            (mov_vector * np.random.rand(len(point))) * random.choice(np.arange(-1, 1.1, 0.1)))).tolist()


def generate_points_inside_hypershape(m_rel: int, n: int, c: list[float], r: float, hyperspheres: bool,
                                      mov_vector: list[float] = None) -> list[list[float]]:
    """
    Populating the beforehand created hypershapes (cubes and spheres) with points (evenly distributed). This function
    populates one given hypershape. So, you have to execute it for every hypershape.

    :param m_rel: The number of relevant features to create. Is used to determine the dimensionality of the points.
    :param n: The number of points to create for this hypershape.
    :param c: The center of the hypershape to populate.
    :param r: The radius/half-edge of the hypershape to populate.
    :param hyperspheres: Whether to populate hyperspheres (True) or hypercubes (False).
    :return: A list of m_rel dimensional points for one hypershape. Every point is a list of m_rel values between -1
        and 1 (and inside of the hypershape).
    """

    xs = []

    for i in range(n):
        x_i = [0] * m_rel

        for j in np.random.permutation(m_rel):
            max_x = c[j] + r
            min_x = c[j] - r
            k = 0
            while True:
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

        if mov_vector is not None:
            x_i = move_point(x_i, mov_vector)

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
    :param random_state: The random state to use. For determinism.
    :return: The pandas.DataFrame dataset extended with the newly created redundant features.
    """

    red = dataset.sample(n=m_red, axis='columns', random_state=random_state)

    for i, column in enumerate(red):
        rand = random.random()
        dataset["red{}".format(i)] = red[column] * (rand if rand > 0 else 1)

    return dataset


def add_irrelevant(dataset: pd.DataFrame, m_irr: int) -> pd.DataFrame:
    """
    Adding irrelevant features. These are columns filled with random values.

    :param dataset: The dataset (containing only the relevant and redundant features) as pandas.DataFrame.
    :param m_irr: The number of irrelevant features to create.
    :return: The pandas.DataFrame dataset extended with the newly created irrelevant features.
    """

    for i in range(m_irr):
        rand_list = np.random.rand(len(dataset))
        rand = random.random()
        dataset["irr{}".format(i)] = (rand_list - rand)

    return dataset


def add_noise_multilabel(labels: pd.DataFrame, noise_levels: Union[list[float], None], q: int) -> list[pd.DataFrame]:
    """
    Create noisy labels for multilabel. The defined noise_levels tell with which probability a point changes a label.

    Example: noise_levels = [0.1] means, that every entry in the One-Hot-Encoded label list switches with a probability
    of 0.1.

    :param labels: The noiseless labels. These are used as ground truth to add noise.
    :param noise_levels: A list with the probability that the labels switch. 3 entries for example cause 3 different
        results.
    :param q: The number of possible labels. This is used for naming the columns of the new noisy labels DataFrame.
    :return: A list containing as many pandas.DataFrame (labels) as there are entries in noise_levels.
    """

    noisy_labels = []

    if noise_levels is not None and noise_levels:
        for i, noise in enumerate(noise_levels):
            new_labels = []
            for j, label in enumerate(labels.values):
                new_values = []
                for k, val in enumerate(label):
                    rand = random.random()
                    if rand < noise:
                        val = (val + 1) % 2
                    new_values.append(val)
                new_labels.append(new_values)
            noisy_labels.append(pd.DataFrame(new_labels, columns=["l{}".format(i) for i in range(q)]))

    return noisy_labels


def add_noise_singlelabel(labels: pd.DataFrame, noise_levels: Union[list[float], None], q: int) -> list[pd.DataFrame]:
    """
    Create noisy labels for singlelabel. The defined noise_levels tell with which probability a point changes its label.

    Example: noise_levels = [0.1] means, that every entry in labels changes its value with a probability of 10% to
    another value.

    :param labels: The noiseless labels. These are used as ground truth to add noise.
    :param noise_levels: A list with the probability that the labels switch. 3 entries for example cause 3 different
        results.
    :param q: The number of possible labels.
    :return: A list containing as many pandas.DataFrame (labels) as there are entries in noise_levels.
    """

    noisy_labels = []
    if noise_levels is not None and noise_levels:
        for i, noise in enumerate(noise_levels):
            new_labels = []
            for label in labels.values:
                rand = random.random()
                if rand < noise:
                    rand = round(random.random() * q)
                    label = (label + rand) % q
                new_labels.append(label)
            noisy_labels.append(pd.DataFrame(new_labels, columns=["labels"]))

    return noisy_labels


def generate(shape: str, m_rel: int, m_irr: int, m_red: int, q: int, n: int, max_r: float = None, min_r: float = None,
             noise_levels: [float] = None, name: str = "Dataset test", random_state: int = None,
             points_distribution: str = None, save_dir: str = None, singlelabel: bool = False,
             iou_threshold: Union[float, list[float]] = None, mov_vectors: list[list[float]] = None) \
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
    :param iou_threshold: The restrictions to apply for the IoU. If None than no restrictions are applied. If a single
        float value, than this is the upper bound of the IoU for every pairing of Hypershapes, so, no Hypershape has an
        IoU larger than this float value with any other Hypershape. If a list of floats, than this list should contain
        exactly 2 values. The first value is the lower bound which enforces every Hypershape to have at least this value
        as IoU with at least one other Hypershape, so, it is possible for two Hypershapes to have no intersection, but
        it ensures that every Hypershape intersects with at least one other Hypershape. The second value is used as
        upper bound and in the same way a single float value would be applied.
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

    random.seed(random_state)
    np.random.seed(random_state)

    hypershapes = generate_small_hypershapes(m_rel, q, max_r, min_r, shape == "spheres", iou_threshold)

    if points_distribution == "uniform":
        ns = [int(n / q)] * q
    else:
        ns = []
        f = n / np.sum(np.array(hypershapes, dtype=object)[:, 0])

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
    for idx, (size, (r, c)) in enumerate(zip(ns, hypershapes)):
        points = generate_points_inside_hypershape(m_rel, size, c, r, shape == "spheres", random.choice(mov_vectors) if mov_vectors is not None else None)
        for point in points:
            if singlelabel:
                point.append(idx)
            dataset.append(point)

    dataset = pd.DataFrame(dataset)

    if not singlelabel:
        labels = assign_labels(dataset, hypershapes, q)
    else:
        labels = dataset[m_rel].to_frame()
        labels.rename({m_rel: "labels"}, axis=1, inplace=True)
        dataset.drop(m_rel, axis=1, inplace=True)

    dataset.rename({i: "rel{}".format(i) for i in range(m_rel)}, axis=1, inplace=True)

    dataset = add_redundant(dataset, m_red, random_state)

    dataset = add_irrelevant(dataset, m_irr)

    dataset = dataset[np.random.permutation(dataset.columns)]

    if not singlelabel:
        noisy_labels = add_noise_multilabel(labels.copy(), noise_levels, q)
    else:
        noisy_labels = add_noise_singlelabel(labels.copy(), noise_levels, q)

    if save_dir:
        save_dir += "/sl" if singlelabel else "/ml"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open("{}/{}_{}_dataset.csv".format(save_dir, name.lower(), shape), "w")as file:
            dataset.to_csv(file, index=False)
        with open("{}/{}_{}_labels.csv".format(save_dir, name.lower(), shape), "w")as file:
            labels.to_csv(file, index=False)
        for ind, noise_level in enumerate(noisy_labels):
            with open("{}/{}_{}_n{}_labels.csv".format(save_dir, name.lower(), shape, ind), "w")as file:
                noise_level.to_csv(file, index=False)

    return dataset, labels, noisy_labels


def plot_sl(dataset, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = dataset["rel0"]
    y = dataset["rel1"]
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(x, y, c=labels.values)
    plt.show()


if __name__ == "__main__":
    # dataset, labels, noisy_labels = generate("cubes", 4, 3, 2, 3, 100, None, None, [0.1, 0.2, 0.5], "Test", 0, None,
    #                                          "ml_datagen")
    #
    # dataset, labels, noisy_labels = generate("spheres", 4, 3, 2, 3, 100, None, None, [], "Test", 0, None,
    #                                          "ml_datagen")
    #
    # dataset, labels, noisy_labels = generate("cubes", 2, 0, 0, 5, 10000, 0.4, 0.2, [0.1, 0.2, 0.5], "Test", 0, None,
    #                                          "ml_datagen", singlelabel=True)
    #
    # dataset, labels, noisy_labels = generate("spheres", 2, 0, 0, 5, 10000, 0.4, 0.2, [], "Test", 0, None,
    #                                          "ml_datagen", singlelabel=True)
    #
    # dataset, labels, noisy_labels = generate("cubes", 2, 0, 0, 5, 10000, 0.4, 0.2, [], "Test", 2, None,
    #                                          "ml_datagen", singlelabel=True)
    #
    # plot_sl(dataset, labels)

    np.random.seed(1)

    dataset, labels, noisy_labels = generate("spheres", 2, 0, 0, 5, 10000, 0.3, 0.1, [], "Test", 1, None,
                                             "ml_datagen", singlelabel=True, iou_threshold=0.3, mov_vectors=np.random.rand(20, 2)*0.4)

    plot_sl(dataset, labels)

    # dataset, labels, noisy_labels = generate("cubes", 2, 0, 0, 5, 10000, 0.4, 0.2, [], "Test", 2, None,
    #                                          "ml_datagen", singlelabel=True, iou_threshold=[0.1, 0.4])
    #
    # plot_sl(dataset, labels)
    #
    # dataset, labels, noisy_labels = generate("spheres", 2, 0, 0, 5, 10000, 0.4, 0.2, [], "Test", 2, None,
    #                                          "ml_datagen", singlelabel=True)
    #
    # plot_sl(dataset, labels)
    #
    # dataset, labels, noisy_labels = generate("spheres", 2, 0, 0, 5, 10000, 0.4, 0.2, [], "Test", 2, None,
    #                                          "ml_datagen", singlelabel=True, iou_threshold=0.3)
    #
    # plot_sl(dataset, labels)
    #
    # dataset, labels, noisy_labels = generate("spheres", 2, 0, 0, 5, 10000, 0.4, 0.2, [], "Test", 2, None,
    #                                          "ml_datagen", singlelabel=True, iou_threshold=[0.1, 0.4])
    #
    # plot_sl(dataset, labels)
