import argparse
import json
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from skmultilearn.adapt import BRkNNaClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
import sklearn.metrics as metrics


def test_dataset(dataset_config: dict):
    n_samples = dataset_config["n_samples"]
    n_features = dataset_config["n_features"]
    n_classes = dataset_config["n_classes"]
    n_labels = dataset_config["n_labels"]
    allow_unlabeled = dataset_config["allow_unlabeled"]
    random_state = dataset_config["random_state"]

    x, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                          n_labels=n_labels, allow_unlabeled=allow_unlabeled, random_state=random_state)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def calc_errors(y_test, y_pred):
    errors = {"f1_score": metrics.f1_score(y_test, y_pred, average='micro'),
              "hamming_loss": metrics.hamming_loss(y_test, y_pred),
              "jaccard_score": metrics.jaccard_score(y_test, y_pred, average="micro"),
              "accuracy_score": metrics.accuracy_score(y_test, y_pred),
              "recall_score": metrics.recall_score(y_test, y_pred, average="micro"),
              "precision_score": metrics.precision_score(y_test, y_pred, average="micro")}
    return errors


def ml_knns(x_train, x_test, y_train, y_test, k: int = None):
    results = {}

    for score in ("f1_weighted", "jaccard_weighted", "accuracy", "precision_weighted", "recall_weighted"):
        results[score] = {}

        for knn in [ml_knn, br_knn]:
            cl_name = str(knn).split(" ")[1]
            results[score][cl_name] = {}
            print("Starting on {} {}".format(score, cl_name))

            classifier_with_k = knn(k, score)
            classifier_without_k = knn(None, score)

            start = time.time()

            classifier_with_k.fit(x_train, y_train)

            time_taken_k = round(time.time() - start, 4)

            print('training time taken with k: ', time_taken_k, 'seconds')
            print('best parameters :', classifier_with_k.best_params_, 'best score: ', classifier_with_k.best_score_)

            y_pred = classifier_with_k.predict(x_test)

            results[score][cl_name]["with_k"] = calc_errors(y_test, y_pred)
            results[score][cl_name]["with_k"]["training_time"] = time_taken_k
            results[score][cl_name]["with_k"]["parameters"] = classifier_with_k.best_params_
            results[score][cl_name]["with_k"]["best_score"] = classifier_with_k.best_score_

            start = time.time()

            classifier_without_k.fit(x_train, y_train)

            time_taken_wo_k = round(time.time() - start, 4)

            print('training time taken without k: ', time_taken_wo_k, 'seconds')
            print('best parameters :', classifier_without_k.best_params_, 'best score: ',
                  classifier_without_k.best_score_)

            y_pred = classifier_without_k.predict(x_test)

            results[score][cl_name]["without_k"] = calc_errors(y_test, y_pred)
            results[score][cl_name]["without_k"]["training_time"] = time_taken_wo_k
            results[score][cl_name]["without_k"]["parameters"] = classifier_without_k.best_params_
            results[score][cl_name]["without_k"]["best_score"] = classifier_without_k.best_score_

    return results


def ml_knn(k, score):
    if k is not None:
        parameters = {'k': [k], 's': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    else:
        parameters = {'k': range(1, 20), 's': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

    return GridSearchCV(MLkNN(), parameters, scoring=score)


def br_knn(k, score):
    if k is not None:
        parameters = {'k': [k]}
    else:
        parameters = {'k': range(1, 20)}

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

            if cl_name == "GradientBoostingClassifier" and alg_name == "label_powerset":
                continue

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
    n_classes = dataset_config["n_classes"]
    x_train, x_test, y_train, y_test = test_dataset(dataset_config)

    # TODO if dataset_config undefined: read from other source

    results_ml_knn = ml_knns(x_train, x_test, y_train, y_test, n_classes)

    results = variants(x_train, x_test, y_train, y_test)

    results["ml_knn"] = results_ml_knn

    results["dataset_information"] = dataset_config

    with open("{}/results.json".format(dir), "w") as file:
        json.dump(results, file)


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
    #     "allow_unlabeled": False,
    #     "random_state": 4
    # }

    dataset_config = {
        "n_samples": 1000,
        "n_features": 2,
        "n_classes": 3,
        "n_labels": 2,
        "allow_unlabeled": False,
        "random_state": 4
    }

    main(args.directory, dataset_config)

# https://xang1234.github.io/multi-label/
# https://link.springer.com/chapter/10.1007/978-3-319-28658-7_43
# http://scikit.ml/api/skmultilearn.cluster.base.html
# https://scikit-learn.org/stable/modules/ensemble.html
