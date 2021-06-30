import argparse
import json
import time
import warnings
from pathlib import Path

from tqdm import tqdm

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


def ml_knns(x_train, x_test, y_train, y_test, k: int = None, debug: bool = False):
    results = {}
    score = "f1_weighted"

    for knn in [ml_knn, br_knn]:
        cl_name = str(knn).split(" ")[1]

        if debug: print("Starting on {}".format(cl_name))

        classifier_with_k = knn(k, score)
        classifier_without_k = knn(None, score)

        name = "{} with_k".format(cl_name)

        if debug: print("Starting on {}".format(name))

        start = time.time()

        classifier_with_k.fit(x_train, y_train)

        time_taken_k = round(time.time() - start, 4)

        if debug: print('training time taken with k: {} seconds'.format(time_taken_k))
        if debug: print('best parameters: {}'.format(classifier_with_k.best_params_))
        if debug: print('best score: {}'.format(classifier_with_k.best_score_))

        y_pred = classifier_with_k.predict(x_test)

        results[name] = calc_errors(y_test, y_pred)
        results[name]["training_time"] = time_taken_k
        results[name]["best_params"] = classifier_with_k.best_params_
        results[name]["best_score"] = classifier_with_k.best_score_
        results[name]["opt_score"] = score

        name = "{} without_k".format(cl_name)

        if debug: print("Starting on {}".format(name))

        start = time.time()

        classifier_without_k.fit(x_train, y_train)

        time_taken_wo_k = round(time.time() - start, 4)

        if debug: print('training time taken without k: {} seconds'.format(time_taken_wo_k))
        if debug: print('best parameters: {}'.format(classifier_without_k.best_params_))
        if debug: print('best score: {}'.format(classifier_without_k.best_score_))

        y_pred = classifier_without_k.predict(x_test)

        results[name] = calc_errors(y_test, y_pred)
        results[name]["training_time"] = time_taken_wo_k
        results[name]["best_params"] = classifier_without_k.best_params_
        results[name]["best_score"] = classifier_without_k.best_score_
        results[name]["opt_score"] = score

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


def binary_relevance(classifier, _):
    return BinaryRelevance(
        classifier=classifier(),
        require_dense=[False, True]
    )


def label_powerset(classifier, _):
    return LabelPowerset(
        classifier=classifier(),
        require_dense=[False, True]
    )


def classifier_chain(classifier, y_size):
    return ClassifierChain(
        classifier=classifier(),
        require_dense=[False, True],
        order=range(y_size)
    )


algorithms = [binary_relevance, label_powerset, classifier_chain]
classifiers = [RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier]


def variants(x_train, x_test, y_train, y_test, debug: bool = False):
    results = {}

    for alg in algorithms:
        alg_name = str(alg).split(" ")[1]

        for cl in classifiers:
            cl_name = str(cl).split(".")[-1].split("'")[0]

            name = "{} {}".format(alg_name, cl_name)

            if name == "label_powerset GradientBoostingClassifier":
                continue

            if debug: print("Starting on {}".format(name))

            start = time.time()

            classifier = alg(cl, y_train.shape[1])

            classifier.fit(x_train, y_train)

            time_taken = round(time.time() - start, 4)

            if debug: print('training time taken: {} seconds'.format(time_taken))

            y_pred = classifier.predict(x_test)

            results[name] = calc_errors(y_test, y_pred)
            results[name]["training_time"] = time_taken

    return results


def _main(dir: str, dataset_config: dict = None, debug: bool = False):
    results = {"type": "multilabel"}
    for rs in tqdm(range(100)):
        n_classes = dataset_config["n_classes"]
        dataset_config["random_state"] = rs
        x_train, x_test, y_train, y_test = test_dataset(dataset_config)

        results[rs] = variants(x_train, x_test, y_train, y_test, debug=debug)
        results_ml_knns = ml_knns(x_train, x_test, y_train, y_test, n_classes, debug=debug)
        results[rs] = {**results[rs], **results_ml_knns}
        results[rs]["dataset_information"] = dataset_config

        with open("{}/results.json".format(dir), "w") as file:
            json.dump(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="multilabel_test_dataset",
                        help='Dictionary where to save images and results.')
    parser.add_argument("--debug", action='store_true', default=False, help='Whether to print debug information.')

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

    Path(args.directory).mkdir(parents=True, exist_ok=True)

    _main(args.directory, dataset_config, args.debug)

# https://xang1234.github.io/multi-label/
# https://link.springer.com/chapter/10.1007/978-3-319-28658-7_43
# http://scikit.ml/api/skmultilearn.cluster.base.html
# https://scikit-learn.org/stable/modules/ensemble.html
