import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def evaluate(data: dict, dir_name: str, plot: bool = False) -> dict:
    helper = {}

    for date in data.values():
        for key, value in date.items():
            if key == "dataset_information": continue
            if key not in helper:
                helper[key] = {}
            for k, v in value.items():
                if v is None: continue
                if k == "opt_score" or k == "best_params": continue
                if k not in helper[key]:
                    helper[key][k] = [v]
                else:
                    helper[key][k].append(v)

    evaluation = {}

    for key, value in helper.items():
        if key not in evaluation:
            evaluation[key] = {}
        for k, v in value.items():
            if plot: plot_boxplot(v, dir_name, "{} {}".format(key, k))
            evaluation[key][k] = {"mean": np.mean(v),
                                  "25th": np.quantile(v, 0.25),
                                  "75th": np.quantile(v, 0.75),
                                  "min": min(v),
                                  "min_ind": int(np.argmin(v)),
                                  "max": max(v),
                                  "max_ind": int(np.argmax(v))}

    return evaluation


def plot_boxplot(data, dir_name: str, title: str):
    Path("{}/plots".format(dir_name)).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data)
    ax.set_xticks([])

    title = title.replace(" ", "_")

    plt.savefig("{}/plots/{}.pdf".format(dir_name, title), format="pdf")
    plt.show()
    plt.close()


def _main(result_path: str, save_path: str, plot: bool = False):
    with open(result_path, "r") as file:
        data = json.load(file)

    data.pop("type", None)

    evaluation = evaluate(data, save_path, plot)

    dataset_information = data["0"]["dataset_information"]
    dataset_information.pop("random_state", None)

    evaluation["dataset_information"] = dataset_information
    evaluation["number_of_datasets"] = len(data)

    with open("{}/evaluation.json".format(save_path), "w") as file:
        json.dump(evaluation, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--result_path", action='store', type=str,
                        default="multilabel_test_dataset/results.json",
                        help='Where the json which should be evaluated is saved.')
    parser.add_argument('-d', "--directory", action='store', type=str, default="multilabel_test_dataset",
                        help='Dictionary where to evaluation results.')
    parser.add_argument("-p", "--plot", action='store_true', default=False, help='Whether to plot the results.')

    args = parser.parse_args()

    _main(args.result_path, args.directory, args.plot)
