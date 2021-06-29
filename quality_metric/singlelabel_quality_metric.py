import argparse
import json
import warnings
from pathlib import Path

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, \
    MiniBatchKMeans, OPTICS, SpectralClustering
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, v_measure_score, hamming_loss, accuracy_score, f1_score, jaccard_score, recall_score, \
    precision_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split

metrics = [adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score,
           homogeneity_score, v_measure_score, hamming_loss, accuracy_score]
metrics_2 = [f1_score, jaccard_score, recall_score, precision_score]


def calculate_errors(true, pred):
    errors = {str(met).split(" ")[1]: met(true, pred) for met in metrics}
    errors_2 = {str(met).split(" ")[1]: met(true, pred, average='micro') for met in metrics_2}

    errors = {**errors, **errors_2}
    return errors


def mean_error(true, pred):
    return np.mean(error for _, error in calculate_errors(true, pred))


def test_dataset(dataset_config: dict, dir_name):
    n_samples = dataset_config["n_samples"]
    n_features = dataset_config["n_features"]
    n_classes = dataset_config["n_classes"]
    n_informative = dataset_config["n_informative"]
    n_redundant = dataset_config["n_redundant"]
    n_clusters_per_class = dataset_config["n_clusters_per_class"]
    random_state = dataset_config["random_state"]

    x, y = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=n_features,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_clusters_per_class=n_clusters_per_class, random_state=random_state)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    plot_clusters(x, y, dir_name, "initial_classification")

    return x, y, x_train, x_test, y_train, y_test


def plot_clusters(X, pred, dir_name: str, title: str):
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    clusters = np.unique(pred)

    for cluster in clusters:
        row_ix = np.where(pred == cluster)

        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        plt.title(title)

    plt.savefig("{}/{}.pdf".format(dir_name, title), format="pdf")
    # plt.show()
    plt.close()


def affinity_propagation(X, y, n_clusters: int = None, score: str = None):
    parameters = {"damping": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]}

    model = GridSearchCV(AffinityPropagation(), parameters, scoring=score)

    model.fit(X, y)

    return model.predict(X), model.best_score_, model.best_params_


def agglomerative(X, y, n_clusters: int, score: str = None):
    model = AgglomerativeClustering(n_clusters=n_clusters)

    return model.fit_predict(X), None, {"n_clusters": n_clusters}


def birch(X, y, n_clusters: int, score: str = None):
    parameters = {"threshold": np.arange(0, 0.51, 0.05), "n_clusters": [n_clusters]}

    model = GridSearchCV(Birch(), parameters, scoring=score)

    model.fit(X, y)

    return model.predict(X), model.best_score_, model.best_params_


def dbscan(X, y, n_clusters=None, score: str = None):
    best_score = 0
    best_params = None
    best_labels = None

    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for min_samples in [6, 9, 12, 15, 18, 21]:
            model = DBSCAN(eps=eps, min_samples=min_samples)

            pred = model.fit_predict(X)

            score = adjusted_rand_score(y, pred)
            if score > best_score:
                best_score = score
                best_labels = pred
                best_params = {"eps": eps, "min_samples": min_samples}

    return best_labels, best_score, best_params


def k_means(X, y, n_clusters, score: str = None):
    model = KMeans(n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X), None, {"n_clusters": n_clusters}


def mini_batch_k_means(X, y, n_clusters, score: str = None):
    model = MiniBatchKMeans(n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X), None, {"n_clusters": n_clusters}


def mean_shift(X, y, n_clusters=None, score: str = None):
    model = MeanShift()

    return model.fit_predict(X), None, None


def optics(X, y, n_clusters=None, score: str = None):
    best_score = 0
    best_params = None
    best_labels = None

    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for min_samples in [6, 9, 12, 15, 18, 21]:
            model = OPTICS(eps=eps, min_samples=min_samples)

            pred = model.fit_predict(X)

            score = adjusted_rand_score(y, pred)
            if score > best_score:
                best_score = score
                best_labels = pred
                best_params = {"eps": eps, "min_samples": min_samples}

    return best_labels, best_score, best_params


def spectral_clustering(X, y, n_clusters, score: str = None):
    model = SpectralClustering(n_clusters=n_clusters)

    return model.fit_predict(X), None, {"n_clusters": n_clusters}


def gaussian_mixture_model(X, y, n_clusters, score: str = None):
    model = GaussianMixture(n_components=n_clusters)

    model.fit(X)

    return model.predict(X), None, {"n_clusters": n_clusters}


clustering_algorithms = [affinity_propagation, agglomerative, birch, dbscan, k_means, mini_batch_k_means, mean_shift,
                         optics, spectral_clustering, gaussian_mixture_model]


def _main(dir: str, dataset_config: dict):
    results = {"type": "singlelabel"}
    for rs in tqdm(range(100)):
        dataset_config["random_state"] = rs
        x, y, x_train, x_test, y_train, y_test = test_dataset(dataset_config, "{}/{}".format(dir, rs))

        n_classes = dataset_config["n_classes"]
        n_clusters_per_class = dataset_config["n_clusters_per_class"]

        num_clusters = n_classes * n_clusters_per_class
        score = "adjusted_rand_score"

        results[rs] = {}

        for alg in clustering_algorithms:
            name = str(alg).split(" ")[1]

            pred, best_score, best_params = alg(x, y, num_clusters, score)

            plot_clusters(x, pred, "{}/{}".format(dir, rs), name)

            errors = calculate_errors(y, pred)

            results[rs][name] = errors
            results[rs][name]["opt_score"] = score
            results[rs][name]["best_score"] = best_score
            results[rs][name]["best_params"] = best_params
            results[rs][name]["expected_clusters"] = num_clusters
            results[rs][name]["found_clusters"] = len(np.unique(pred))

        results[rs]["dataset_information"] = dataset_config

        with open("{}/results.json".format(dir), "w") as file:
            json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="singlelabel_test_dataset",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    dataset_config = {
        "n_samples": 10000,
        "n_features": 2,
        "n_classes": 4,
        "n_informative": 2,
        "n_redundant": 0,
        "n_clusters_per_class": 1,
        "random_state": 4
    }

    _main(args.directory, dataset_config)

# TODO: Classification/Clustering Abgrenzung und Untersuchung

# https://machinelearningmastery.com/clustering-algorithms-with-python/#:~:text=Cluster%20analysis%2C%20or%20clustering%2C%20is,or%20clusters%20in%20feature%20space.
# https://scikit-learn.org/stable/modules/model_evaluation.html

# https://research.aimultiple.com/synthetic-data-generation/
# https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae?gi=42a2bf0e9d95
# https://research.aimultiple.com/synthetic-data/
# https://mostly.ai/
# https://paperswithcode.com/task/synthetic-data-generation
# https://www.kdnuggets.com/2021/02/overview-synthetic-data-types-generation-methods.html
# https://stackabuse.com/generating-synthetic-data-with-numpy-and-scikit-learn/
# https://github.com/sdv-dev/SDV
# https://www.statice.ai/post/how-generate-synthetic-data
# https://en.wikipedia.org/wiki/Synthetic_data
