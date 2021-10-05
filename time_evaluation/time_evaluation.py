import timeit
import json

from sklearn.datasets import make_classification, make_blobs
from generators.ml_datagen import generate


if __name__ == "__main__":
    timings = {"csm_1000_2d_4c": min(timeit.repeat(
        'generate(n_samples=1000, shapes="mix", m_rel=2, m_irr=0, m_red=0, n_classes=4, mov_vectors="random", random_state=0)',
        "from __main__ import generate", repeat=10, number=100)),
        "blobs_1000_2d_4c": min(timeit.repeat('make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)',
                                              "from __main__ import make_blobs", repeat=10, number=100)),
        "mc_1000_2d_4c": min(timeit.repeat(
            'make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=4, random_state=0, n_clusters_per_class=1)',
            "from __main__ import make_classification", repeat=10, number=100)),
        "csm_1000_10d_4c": min(timeit.repeat(
            'generate(n_samples=1000, shapes="mix", m_rel=10, m_irr=0, m_red=0, n_classes=4, mov_vectors="random", random_state=0)',
            "from __main__ import generate", repeat=10, number=100)),
        "blobs_1000_10d_4c": min(timeit.repeat('make_blobs(n_samples=1000, n_features=10, centers=4, random_state=0)',
                                               "from __main__ import make_blobs", repeat=10, number=100)),
        "mc_1000_10d_4c": min(timeit.repeat(
            'make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_classes=4, random_state=0, n_clusters_per_class=1)',
            "from __main__ import make_classification", repeat=10, number=100)),
        "csm_1000_20d_4c": min(timeit.repeat(
            'generate(n_samples=1000, shapes="mix", m_rel=20, m_irr=0, m_red=0, n_classes=4, mov_vectors="random", random_state=0)',
            "from __main__ import generate", repeat=10, number=100)),
        "blobs_1000_20d_4c": min(timeit.repeat('make_blobs(n_samples=1000, n_features=20, centers=4, random_state=0)',
                                              "from __main__ import make_blobs", repeat=10, number=100)),
        "mc_1000_20d_4c": min(timeit.repeat(
            'make_classification(n_samples=1000, n_features=20, n_informative=20, n_redundant=0, n_classes=4, random_state=0, n_clusters_per_class=1)',
            "from __main__ import make_classification", repeat=10, number=100)),
        "csm_10000_2d_4c": min(timeit.repeat(
            'generate(n_samples=10000, shapes="mix", m_rel=2, m_irr=0, m_red=0, n_classes=4, mov_vectors="random", random_state=0)',
            "from __main__ import generate", repeat=10, number=100)),
        "blobs_10000_2d_4c": min(timeit.repeat('make_blobs(n_samples=10000, n_features=2, centers=4, random_state=0)',
                                              "from __main__ import make_blobs", repeat=10, number=100)),
        "mc_10000_2d_4c": min(timeit.repeat(
            'make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_classes=4, random_state=0, n_clusters_per_class=1)',
            "from __main__ import make_classification", repeat=10, number=100)),
    }

    with open("timings.json", "w") as file:
        json.dump(timings, file)
