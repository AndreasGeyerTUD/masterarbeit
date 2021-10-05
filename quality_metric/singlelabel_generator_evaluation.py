import json
import time
import warnings
from functools import partial
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, jaccard_score, hamming_loss
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, v_measure_score

from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, \
    MiniBatchKMeans, OPTICS, SpectralClustering

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import multiprocessing as mp
from scipy.optimize import linear_sum_assignment as linear_assignment
from generators.csm_datagen import generate

clustering_metrics = [adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score,
                      homogeneity_score, v_measure_score]
classification_metrics = [accuracy_score, recall_score, precision_score, f1_score, jaccard_score, hamming_loss]


def plot_clusters(X, pred, dir_name: str, title: str):
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    clusters = np.unique(pred)

    for cluster in clusters:
        row_ix = np.where(pred == cluster)

        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        plt.title(title)

    plt.savefig("{}/{}.png".format(dir_name, title), format="png")
    # plt.show()
    plt.close()


def error_with_lp(y_true, y_pred, metric):
    """
    Permute labels of y_pred to match y_true as much as possible

    https://programtalk.com/python-examples/sklearn.utils.linear_assignment_.linear_assignment/
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    label1 = np.unique(y_true)
    n_class1 = len(label1)

    label2 = np.unique(y_pred)
    n_class2 = len(label2)

    if n_class1 != n_class2: return metric(y_true, y_pred)

    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))

    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = y_true == label1[i]
            tt = y_pred == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)

    A = linear_assignment(-G)

    new_l2 = np.zeros(y_pred.shape)
    for i in range(0, n_class2):
        new_l2[y_pred == label2[A[1][i]]] = label1[A[0][i]]

    new_y_pred = new_l2.astype(int)

    return metric(y_true, new_y_pred)


def calc_clustering_metrics(y_true, y_pred):
    return {str(met).split(" ")[1]: met(y_true, y_pred) for met in clustering_metrics}


def calc_classification_metrics_clust(y_true, y_pred):
    return {
        "f1_score": error_with_lp(y_true, y_pred, partial(f1_score, average="macro", zero_division=0)),
        "hamming_loss": error_with_lp(y_true, y_pred, hamming_loss),
        "jaccard_score": error_with_lp(y_true, y_pred, partial(jaccard_score, average="macro", zero_division=0)),
        "accuracy_score": error_with_lp(y_true, y_pred, accuracy_score),
        "recall_score": error_with_lp(y_true, y_pred, partial(recall_score, average="macro", zero_division=0)),
        "precision_score": error_with_lp(y_true, y_pred, partial(precision_score, average="macro", zero_division=0))
    }


def calc_classification_metrics_class(y_true, y_pred):
    return {"f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "jaccard_score": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy_score": accuracy_score(y_true, y_pred),
            "recall_score": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_score": precision_score(y_true, y_pred, average="macro", zero_division=0)}


def affinity_propagation(X: np.ndarray, n_clusters: int = None):
    model = AffinityPropagation()

    model.fit(X)

    return model.predict(X)


def agglomerative(X: np.ndarray, n_clusters: int):
    model = AgglomerativeClustering(n_clusters=n_clusters)

    return model.fit_predict(X)


def birch(X: np.ndarray, n_clusters: int = None):
    model = Birch()

    model.fit(X)

    return model.predict(X)


def dbscan(X: np.ndarray, n_clusters: int = None):
    model = DBSCAN()

    return model.fit_predict(X)


def k_means(X: np.ndarray, n_clusters: int):
    model = KMeans(n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X)


def mini_batch_k_means(X: np.ndarray, n_clusters: int):
    model = MiniBatchKMeans(n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X)


def mean_shift(X: np.ndarray, n_clusters: int = None):
    model = MeanShift()

    return model.fit_predict(X)


def optics(X: np.ndarray, n_clusters: int = None):
    model = OPTICS()

    return model.fit_predict(X)


def spectral_clustering(X: np.ndarray, n_clusters: int):
    model = SpectralClustering(n_clusters=n_clusters)

    return model.fit_predict(X)


def gaussian_mixture_model(X: np.ndarray, n_clusters: int):
    model = GaussianMixture(n_components=n_clusters)

    model.fit(X)

    return model.predict(X)


clustering_algorithms = [affinity_propagation, agglomerative, birch, dbscan, k_means, mini_batch_k_means, mean_shift,
                         optics, spectral_clustering, gaussian_mixture_model]

classification_algorithms = [KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LinearSVC,
                             LogisticRegression, GaussianNB]


def generate_random_dataset(i: int):
    if i < 100:
        dataset, labels = make_classification(n_samples=1000, n_features=2, n_classes=4, n_informative=2,
                                              n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=i)
    elif i < 200:
        dataset, labels, _ = generate(n_samples=1000, m_rel=2, m_irr=0, m_red=0, n_classes=4, n_clusters_per_class=1,
                                      shapes="mix", random_state=i, singlelabel=True, mov_vectors="random")
        dataset = dataset.to_numpy()
        labels = labels.to_numpy().reshape(len(labels))
    elif i < 300:
        dataset, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=i)
    elif i < 400:
        dataset, labels = make_classification(n_samples=1000, n_features=5, n_classes=4, n_informative=5,
                                              n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=i)
    elif i < 500:
        dataset, labels, _ = generate(n_samples=1000, m_rel=5, m_irr=0, m_red=0, n_classes=4, n_clusters_per_class=1,
                                      shapes="mix", random_state=i, singlelabel=True, mov_vectors="random")
        dataset = dataset.to_numpy()
        labels = labels.to_numpy().reshape(len(labels))
    elif i < 600:
        dataset, labels = make_blobs(n_samples=1000, centers=4, n_features=5, random_state=i)
    elif i < 700:
        dataset, labels = make_classification(n_samples=5000, n_features=2, n_classes=4, n_informative=2,
                                              n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=i)
    elif i < 800:
        dataset, labels, _ = generate(n_samples=5000, m_rel=2, m_irr=0, m_red=0, n_classes=4, n_clusters_per_class=1,
                                      shapes="mix", random_state=i, singlelabel=True, mov_vectors="random")
        dataset = dataset.to_numpy()
        labels = labels.to_numpy().reshape(len(labels))
    elif i < 900:
        dataset, labels = make_blobs(n_samples=5000, centers=4, n_features=2, random_state=i)
    elif i < 1000:
        dataset, labels = make_classification(n_samples=5000, n_features=5, n_classes=4, n_informative=5,
                                              n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=i)
    elif i < 1100:
        dataset, labels, _ = generate(n_samples=5000, m_rel=5, m_irr=0, m_red=0, n_classes=4, n_clusters_per_class=1,
                                      shapes="mix", random_state=i, singlelabel=True, mov_vectors="random")
        dataset = dataset.to_numpy()
        labels = labels.to_numpy().reshape(len(labels))
    elif i < 1200:
        dataset, labels = make_blobs(n_samples=5000, centers=4, n_features=5, random_state=i)

    return dataset, labels


def clustering(x, y, dir_name: str):
    n_clusters = len(np.unique(y))

    results = {}

    for alg in clustering_algorithms:
        name = str(alg).split(" ")[1]

        start = time.time()
        pred = alg(x, n_clusters)
        time_taken = round(time.time() - start, 4)

        results[name] = {"clustering": calc_clustering_metrics(y, pred),
                         "classification": calc_classification_metrics_clust(y, pred),
                         "found_clusters": len(np.unique(pred)),
                         "expected_clusters": n_clusters,
                         "time": time_taken}

        if x.shape[1] == 2:
            plot_clusters(x, pred, dir_name, name)

    return results


def classification(x, y, dir_name: str):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    results = {}

    for clas in classification_algorithms:
        name = str(clas).split(".")[-1][:-2]

        start = time.time()

        model = clas()

        model.fit(x_train, y_train)

        pred = model.predict(x_test)

        time_taken = round(time.time() - start, 4)

        results[name] = {"clustering": calc_clustering_metrics(y_test, pred),
                         "classification": calc_classification_metrics_clust(y_test, pred),
                         "time": time_taken}

        whole_pred = model.predict(x)

        if x.shape[1] == 2:
            plot_clusters(x, whole_pred, dir_name, name)

    return results


def _main(l: list):
    results = {}
    for i in tqdm(l):
        dataset, labels = generate_random_dataset(i)
        if dataset.shape[1] == 2:
            plot_clusters(dataset, labels, "evaluation/{}".format(i), "initial_clustering")

        results["clustering_alg"] = clustering(dataset, labels, "evaluation/{}".format(i))

        results["classification_alg"] = classification(dataset, labels, "evaluation/{}".format(i))

        Path("evaluation/{}".format(i)).mkdir(parents=True, exist_ok=True)
        with open("{}/results.json".format("evaluation/{}".format(i)), "w") as file:
            json.dump(results, file)


def split_list(a_list, wanted_parts=1):
    length = len(a_list)
    return [a_list[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


if __name__ == "__main__":
    cores = 12

    pool = mp.Pool(cores)
    pool.map(_main, split_list(list(range(300, 600)) + list(range(900, 1200)), cores))
