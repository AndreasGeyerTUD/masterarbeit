import argparse
import json
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# import networkx as nx
# from matplotlib import pyplot as plt
import sklearn.model_selection
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import BRkNNaClassifier
from skmultilearn.adapt import MLkNN
# from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
# from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
# from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
import sklearn.metrics as metrics


def test_dataset(n_samples, n_features, n_classes, n_labels, allow_unlabeled):
    x, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                          n_labels=n_labels, allow_unlabeled=allow_unlabeled)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    # plot_clusters(X, y, "test_dataset", "initial_classification")

    return x_train, x_test, y_train, y_test


# def plot_clusters(X, pred, dir_name: str, title: str):
#     # min_x = np.min(X[:, 0])
#     # max_x = np.max(X[:, 0])
#     #
#     # min_y = np.min(X[:, 1])
#     # max_y = np.max(X[:, 1])
#     #
#     # classif = OneVsRestClassifier(SVC(kernel='linear'))
#     # classif.fit(X, Y)
#     #
#     # plt.subplot(2, 2, subplot)
#     # plt.title(title)
#     #
#     # zero_class = np.where(Y[:, 0])
#     # one_class = np.where(Y[:, 1])
#     # plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
#     # plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
#     #             facecolors='none', linewidths=2, label='Class 1')
#     # plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
#     #             facecolors='none', linewidths=2, label='Class 2')
#     #
#     # plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
#     #                 'Boundary\nfor class 1')
#     # plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
#     #                 'Boundary\nfor class 2')
#     # plt.xticks(())
#     # plt.yticks(())
#     #
#     # plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
#     # plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
#     # if subplot == 2:
#     #     plt.xlabel('First principal component')
#     #     plt.ylabel('Second principal component')
#     #     plt.legend(loc="upper left")
#
#
#     clusters = np.unique(pred)
#
#     for cluster in clusters:
#         row_ix = np.where(pred == cluster)
#
#         plt.scatter(X[row_ix, 0], X[row_ix, 1])
#         plt.title(title)
#
#     plt.savefig("{}/{}.pdf".format(dir_name, title), format="pdf")
#     plt.show()
#
#
# def label_graph(X, y, feature_names=None, label_names=None):
#     graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
#
#     label_names = range(y.shape[1]) if label_names is None else label_names
#     edge_map = graph_builder.transform(y)
#     print("{} labels, {} edges".format(len(label_names), len(edge_map)))
#
#     def to_membership_vector(partition):
#         return {
#             member: partition_id
#             for partition_id, members in enumerate(partition)
#             for member in members
#         }
#
#     clusterer = NetworkXLabelGraphClusterer(graph_builder, method='louvain')
#
#     partition = clusterer.fit_predict(X, y)
#     membership_vector = to_membership_vector(partition)
#     print('There are', len(partition), 'clusters')
#
#     names_dict = dict(enumerate(x for x in label_names))
#
#     nx.draw(
#         clusterer.graph_,
#         pos=nx.spring_layout(clusterer.graph_, k=4),
#         labels=names_dict,
#         with_labels=True,
#         width=[10 * x / y.shape[0] for x in clusterer.weights_['weight']],
#         node_color=[membership_vector[i] for i in range(y.shape[1])],
#         cmap=plt.cm.viridis,
#         node_size=250,
#         font_size=10,
#         font_color='white',
#         alpha=0.8
#     )
#
#     plt.show()


def calc_errors(y_test, y_pred):
    errors = {"f1_score": metrics.f1_score(y_test, y_pred, average='micro'),
              "hamming_loss": metrics.hamming_loss(y_test, y_pred),
              "jaccard_score": metrics.jaccard_score(y_test, y_pred, average="micro"),
              "accuracy_score": metrics.accuracy_score(y_test, y_pred),
              "recall_score": metrics.recall_score(y_test, y_pred, average="micro"),
              "precision_score": metrics.precision_score(y_test, y_pred, average="micro")}
    return errors


def ml_knns(x_train, x_test, y_train, y_test):
    results = {}

    for knn in [ml_knn, br_knn]:
        cl_name = str(knn).split(" ")[1]
        print("Starting on " + cl_name)

        start = time.time()

        # TODO: k nochmal prüfen
        classifier = knn(1)

        classifier.fit(x_train, y_train)

        time_taken = round(time.time() - start, 4)

        print('training time taken: ', time_taken, 'seconds')
        print('best parameters :', classifier.best_params_, 'best score: ', classifier.best_score_)

        y_pred = classifier.predict(x_test)

        results[cl_name] = calc_errors(y_test, y_pred)
        results[cl_name]["training_time"] = time_taken

    return results


def ml_knn(k):
    parameters = {'k': range(1, 3), 's': [0.5, 0.7, 1.0]}
    score = 'f1_micro'

    return GridSearchCV(MLkNN(), parameters, scoring=score)


def br_knn(k):
    parameters = {'k': range(3, 5)}
    score = 'f1_micro'

    return GridSearchCV(BRkNNaClassifier(), parameters, scoring=score)


def binary_relevance(classifier, y):
    return BinaryRelevance(
        classifier=classifier(),
        require_dense=[False, True]
    )


def label_powerset(classifier, y):
    return LabelPowerset(
        classifier=classifier(),
        require_dense=[False, True]
    )


def classifier_chain(classifier, y):
    return ClassifierChain(
        classifier=classifier(),
        require_dense=[False, True],
        order=range(y.shape[1])
    )


algorithms = [binary_relevance, label_powerset, classifier_chain]
classifiers = [RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier]


def variants(x_train, x_test, y_train, y_test):
    results = {}

    for alg in algorithms:
        alg_name = str(alg).split(" ")[1]

        results[alg_name] = {}

        for cl in classifiers:
            cl_name = str(cl).split(".")[-1].split("'")[0]

            print("Starting on " + alg_name + " " + cl_name)

            start = time.time()

            classifier = alg(cl, y_train)

            classifier.fit(x_train, y_train)

            time_taken = round(time.time() - start, 4)

            print('training time taken: ', time_taken, 'seconds')

            y_pred = classifier.predict(x_test)

            results[alg_name][cl_name] = calc_errors(y_test, y_pred)
            results[alg_name][cl_name]["training_time"] = time_taken

    return results


def main(dir: str, dataset_config: dict = None):
    n_samples = dataset_config["n_samples"]
    n_features = dataset_config["n_features"]
    n_classes = dataset_config["n_classes"]
    n_labels = dataset_config["n_labels"]
    allow_unlabeled = dataset_config["allow_unlabeled"]
    x_train, x_test, y_train, y_test = test_dataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                                    n_labels=n_labels, allow_unlabeled=allow_unlabeled)

    # TODO if dataset_config undefined: read from other source

    results_ml_knn = ml_knns(x_train, x_test, y_train, y_test)

    results = variants(x_train, x_test, y_train, y_test)

    results["ml_knn"] = results_ml_knn

    results["dataset_information"] = dataset_config

    with open("{}/results.json".format(dir), "w") as file:
        json.dump(results, file)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="multilabel_test_dataset",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    # dataset_config = {
    #     "n_samples": 10000,
    #     "n_features": 21,
    #     "n_classes": 12,
    #     "n_labels": 4,
    #     "allow_unlabeled": False
    # }

    dataset_config = {
        "n_samples": 1000,
        "n_features": 2,
        "n_classes": 3,
        "n_labels": 2,
        "allow_unlabeled": False
    }

    main(args.directory, dataset_config)

# https://xang1234.github.io/multi-label/
# https://link.springer.com/chapter/10.1007/978-3-319-28658-7_43
# http://scikit.ml/api/skmultilearn.cluster.base.html
# https://scikit-learn.org/stable/modules/ensemble.html
