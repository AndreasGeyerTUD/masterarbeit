import argparse
import json

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, \
    MiniBatchKMeans, OPTICS, SpectralClustering
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, \
    homogeneity_score, v_measure_score
from sklearn.mixture import GaussianMixture

metrics = [adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score,
           homogeneity_score, v_measure_score]


def calculate_errors(true, pred):
    errors = list([(str(met).split(" ")[1], met(true, pred)) for met in metrics])
    return errors


def mean_error(true, pred):
    return np.mean(error for _, error in calculate_errors(true, pred))


def test_dataset():
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               random_state=4)

    plot_clusters(X, y, "test_dataset", "initial_classification")

    return X, y


def plot_clusters(X, pred, dir_name: str, title: str):
    clusters = np.unique(pred)

    for cluster in clusters:
        row_ix = np.where(pred == cluster)

        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        plt.title(title)

    plt.savefig("{}/{}.pdf".format(dir_name, title), format="pdf")
    plt.show()


def affinity_propagation(X, n_clusters=None):
    model = AffinityPropagation(damping=0.9)

    model.fit(X)

    return model.predict(X)


def agglomerative(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)

    return model.fit_predict(X)


def birch(X, n_clusters):
    model = Birch(threshold=0.01, n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X)


def dbscan(X, n_clusters=None):
    model = DBSCAN(eps=0.30, min_samples=9)

    return model.fit_predict(X)


def k_means(X, n_clusters):
    model = KMeans(n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X)


def mini_batch_k_means(X, n_clusters):
    model = MiniBatchKMeans(n_clusters=n_clusters)

    model.fit(X)

    return model.predict(X)


def mean_shift(X, n_clusters=None):
    model = MeanShift()

    return model.fit_predict(X)


def optics(X, n_clusters=None):
    model = OPTICS(eps=0.8, min_samples=10)

    return model.fit_predict(X)


def spectral_clustering(X, n_clusters):
    model = SpectralClustering(n_clusters=n_clusters)

    return model.fit_predict(X)


def gaussian_mixture_model(X, n_clusters):
    model = GaussianMixture(n_components=n_clusters)

    model.fit(X)

    return model.predict(X)


clustering_algorithms = [affinity_propagation, agglomerative, birch, dbscan, k_means, mini_batch_k_means, mean_shift,
                         optics, spectral_clustering, gaussian_mixture_model]


def _main(X, y, dir: str):
    num_clusters = len(np.unique(y))

    results = {}

    for alg in clustering_algorithms:
        name = str(alg).split(" ")[1]

        pred = alg(X, num_clusters)

        plot_clusters(X, pred, dir, name)

        errors = calculate_errors(y, pred)

        results[name] = errors

    print(results)

    with open("{}/results.json".format(dir), "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="test_dataset",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    X, y = test_dataset()

    _main(X, y, args.directory)


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
