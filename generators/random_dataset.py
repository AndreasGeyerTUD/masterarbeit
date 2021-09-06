import datetime
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
import ml_datagen


def random_ml_datagen(random_state: int, path: str = None):
    random.seed(random_state)
    np.random.seed(random_state)

    rs = random.randint(0, 100000)
    n_samples = random.randrange(100, 50001, 50)
    n_features = random.randrange(2, 41, 1)
    if int(n_features / 3) < 1:
        n_redundant = 0
        n_repeated = 0
    else:
        n_redundant = random.randrange(0, int(n_features / 3), 1)
        n_repeated = random.randrange(0, int(n_features / 3), 1)
    n_informative = n_features - n_redundant - n_repeated
    n_classes = random.randrange(2, 21, 1)

    shape = random.choice(["cubes", "spheres"])

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
                "random_state": rs
            }}
            print(dic, file=save_file)

    dataset, labels, _ = ml_datagen.generate(shapes=shape, m_rel=n_informative, m_irr=n_repeated, m_red=n_redundant,
                                             n_classes=n_classes, n_samples=n_samples, singlelabel=True, random_state=rs)

    return dataset, labels


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


if __name__ == "__main__":
    # ts = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    #
    # for i in tqdm(range(1000)):
    #     dataset, labels = random_sklearn(i, ts + "/ds.txt")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i in tqdm(range(1000)):
        dataset, labels = random_ml_datagen(i, ts + "/ds.txt")
