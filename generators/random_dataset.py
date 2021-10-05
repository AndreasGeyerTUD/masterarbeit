import datetime
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
import csm_datagen
import multiprocessing as mp


def exp_1_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = "cubes"

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "mov_vectors": mov_vectors,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              mov_vectors=mov_vectors)

    return dataset, labels


def exp_2_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = "mix"

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "mov_vectors": mov_vectors,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              mov_vectors=mov_vectors)

    return dataset, labels


def exp_3_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = [("cubes", 0.6), ("spheres", 0.2), ("moons", 0.2)]

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "mov_vectors": mov_vectors,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              mov_vectors=mov_vectors)

    return dataset, labels


def exp_4_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = int(np.random.normal(15000, 6000, 1)[0])
    n_samples = n_samples if n_samples > 100 else 100
    n_informative = int(np.random.normal(20, 10, 1)[0])
    n_informative = n_informative if n_informative > 3 else 3
    n_redundant = random.randrange(0, 11, 1)
    n_repeated = random.randrange(0, 11, 1)
    n_features = n_informative + n_redundant + n_repeated
    n_classes = random.randrange(2, 15, 1)
    n_clusters_per_class = random.randrange(1, 6, 1)

    n_categorical_variables = random.randrange(0, 11, 1)
    n_categorical_variables = n_categorical_variables if n_categorical_variables <= n_informative else n_informative
    categorical_variables = []
    for i in range(n_categorical_variables):
        v = int(np.random.normal(4, 6, 1)[0])
        v = v if v > 1 else 2
        categorical_variables.append(v)

    shape = "mix"

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "_n_features": n_features,
                "categorical_variables": categorical_variables,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              mov_vectors="random", categorical_variables=categorical_variables)

    return dataset, labels


def exp_5_random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = int(np.random.normal(15000, 6000, 1)[0])
    n_samples = n_samples if n_samples > 100 else 100
    n_informative = int(np.random.normal(20, 10, 1)[0])
    n_informative = n_informative if n_informative > 3 else 3
    n_redundant = random.randrange(0, 11, 1)
    n_repeated = random.randrange(0, 11, 1)
    n_features = n_informative + n_redundant + n_repeated
    n_classes = random.randrange(2, 15, 1)
    n_clusters_per_class = random.randrange(1, 6, 1)

    n_categorical_variables = random.randrange(0, 11, 1)
    n_categorical_variables = n_categorical_variables if n_categorical_variables <= n_informative else n_informative
    categorical_variables = []
    for i in range(n_categorical_variables):
        v = int(np.random.normal(4, 6, 1)[0])
        v = v if v > 1 else 2
        categorical_variables.append(v)

    shape = "mix"

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "_n_features": n_features,
                "categorical_variables": categorical_variables,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "random_state": rs
            }}
            print(dic, file=save_file)

    if random.random() >= 0.5:
        dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                                  n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                                  random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                                  mov_vectors="random", categorical_variables=categorical_variables)
    else:
        dataset, labels = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                              n_redundant=n_redundant, n_repeated=n_repeated,
                                              n_clusters_per_class=n_clusters_per_class, random_state=rs)

    return dataset, labels


def exp_6_random_ml_datagen(random_state: int, path: str = None):
    # Executed but not in thesis as same result
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    mov_vectors = np.random.rand(20, n_informative)

    shape = random.choice(["spheres", "cubes", "moons", "mix"])

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "mov_vectors": mov_vectors,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              mov_vectors=mov_vectors)

    return dataset, labels


def exp_7_random_ml_datagen(random_state: int, path: str = None):
    # Executed but not in thesis as same result
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    n_random_points = random.random() * 0.2
    mov_vectors = np.random.rand(20, n_informative)

    shape = "mix"

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "n_random_points": n_random_points,
                "mov_vectors": mov_vectors,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              random_points=n_random_points, mov_vectors=mov_vectors)

    return dataset, labels


def exp_8_random_ml_datagen(random_state: int, path: str = None):
    # Executed but not in thesis as same result
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = int(np.random.normal(15000, 6000, 1)[0])
    n_samples = n_samples if n_samples > 100 else 100
    n_informative = int(np.random.normal(20, 10, 1)[0])
    n_informative = n_informative if n_informative > 3 else 3
    n_redundant = random.randrange(0, 11, 1)
    n_repeated = random.randrange(0, 11, 1)
    n_features = n_informative + n_redundant + n_repeated
    n_classes = random.randrange(2, 15, 1)
    n_clusters_per_class = random.randrange(1, 6, 1)

    shape = [("cubes", 0.6), ("spheres", 0.2), ("moons", 0.2)]

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              mov_vectors="random")

    return dataset, labels


def exp_9_random_ml_datagen(random_state: int, path: str = None):
    # Executed but not in thesis as same result
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)
    n_random_points = random.random() * 0.2
    mov_vectors = np.random.rand(20, n_informative)

    n_categorical_variables = random.randrange(0, 11, 1)
    n_categorical_variables = n_categorical_variables if n_categorical_variables <= n_informative else n_informative
    categorical_variables = []
    for i in range(n_categorical_variables):
        v = int(np.random.normal(4, 6, 1)[0])
        v = v if v > 1 else 2
        categorical_variables.append(v)

    shape = "mix"

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "shape": shape,
                "n_samples": n_samples,
                "__n_features": n_features,
                "categorical_variables": categorical_variables,
                "m_rel": n_informative,
                "m_red": n_redundant,
                "m_irr": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "n_random_points": n_random_points,
                "mov_vectors": mov_vectors,
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = csm_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                              n_classes=n_classes, n_samples=n_samples, singlelabel=True,
                                              random_state=rs, n_clusters_per_class=n_clusters_per_class,
                                              random_points=n_random_points, mov_vectors=mov_vectors,
                                              categorical_variables=categorical_variables)


def random_sklearn(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 61, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)
    n_clusters_per_class = random.randrange(1, 11, 1)

    while n_classes * n_clusters_per_class > 2 ** n_informative:
        if n_clusters_per_class > 1:
            n_clusters_per_class -= 1
        elif n_classes > 2:
            n_classes -= 1
        else:
            n_informative += 1

    if path is not None:
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "a+") as save_file:
            dic = {random_state: {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_informative": n_informative,
                "n_redundant": n_redundant,
                "n_repeated": n_repeated,
                "n_classes": n_classes,
                "n_clusters_per_class": n_clusters_per_class,
                "random_state": rs
            }}
            print(dic, file=save_file)

    return make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class, random_state=rs)
