import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def evaluate(data: list, dir_name: str, plot: bool = False) -> Tuple[dict, pd.DataFrame]:
    helper = {}

    for dataset in data:
        for k1, v1 in dataset.items():
            if k1 not in helper:
                helper[k1] = {}

            for k2, v2 in v1.items():
                if k2 not in helper[k1]:
                    helper[k1][k2] = {}

                for k3, v3 in v2.items():
                    if isinstance(v3, dict):
                        if k3 not in helper[k1][k2]:
                            helper[k1][k2][k3] = {}
                        for k4, v4 in v3.items():
                            if k4 not in helper[k1][k2][k3]:
                                helper[k1][k2][k3][k4] = [v4]
                            else:
                                helper[k1][k2][k3][k4].append(v4)
                    else:
                        if k3 not in helper[k1][k2]:
                            helper[k1][k2][k3] = [v3]
                        else:
                            helper[k1][k2][k3].append(v3)

    evaluation = {}
    latex = pd.DataFrame([])

    for k1, v1 in helper.items():
        if k1 not in evaluation:
            evaluation[k1] = {}

        for k2, v2 in v1.items():
            if k2 not in evaluation[k1]:
                evaluation[k1][k2] = {}

            for k3, v3 in v2.items():
                if isinstance(v3, dict):
                    if k3 not in evaluation[k1][k2]:
                        evaluation[k1][k2][k3] = {}
                    for k4, v4 in v3.items():
                        if plot: plot_boxplot(v4, dir_name, "{} {}".format(k2, k4))
                        evaluation[k1][k2][k3][k4] = box_plot_dict(v4)
                        latex.loc[k2, k4] = round(np.mean(v4), 4)
                else:
                    if plot: plot_boxplot(v3, dir_name, "{} {}".format(k2, k3))
                    evaluation[k1][k2][k3] = box_plot_dict(v3)
                    if k3 != "expected_clusters":
                        latex.loc[k2, k3] = round(np.mean(v3), 4)

    return evaluation, latex


def box_plot_dict(values: list):
    return {"mean": round(np.mean(values), 4),
            "25th": round(np.quantile(values, 0.25), 4),
            "75th": round(np.quantile(values, 0.75), 4),
            "min": round(min(values), 4),
            "min_ind": int(np.argmin(values)),
            "max": round(max(values), 4),
            "max_ind": int(np.argmax(values))}


def plot_boxplot(data, dir_name: str, title: str):
    Path("{}/result_plots".format(dir_name)).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data)
    ax.set_xticks([])

    title = title.replace(" ", "_")

    plt.savefig("{}/result_plots/{}.pdf".format(dir_name, title), format="pdf")
    # plt.show()
    plt.close()


def _main(result_path: str, save_path: str, plot: bool = False):
    results_list = []
    j = 0

    for i in tqdm(range(1200)):
        path = Path(result_path).joinpath(str(i)).joinpath("results.json")

        with open(path, "r") as file:
            results_list.append(json.load(file))

        if i % 100 == 99:
            evaluation, latex = evaluate(results_list, save_path + "/box_plots_{}".format(j), plot)

            with open("{}/evaluation_{}.json".format(save_path, j), "w") as file:
                json.dump(evaluation, file)
            with open("{}/latex_{}.txt".format(save_path, j), "w") as file:
                latex.columns = ["AMI", "ARI", "Comp", "FM", "Hom", "V-Mea", "F1", "Ham", "Jac", "Acc", "Rec", "Pre",
                                 "# clus", "time (s)"]
                new_index = ["Affinity Propagation", "Agglomerative Clust", "BIRCH", "DBScan", "K-Means",
                             "Mini-Batch K-Means", "Mean-Shift", "OPTICS", "Spectral Clustering", "Gaussian Mixture",
                             "KNN", "Decision Tree", "Random Forest", "SVM", "Logistic Regression", "Naive Bayes"]
                latex.rename(index={l1: l2 for l1, l2 in zip(latex.index, new_index)}, inplace=True)
                latex.to_latex(file, header=True, index=True, caption=str(j), label="tabel:{}".format(j),
                               column_format="lrrrrrrrrrrrrrr", position="tb", na_rep="-")

            j += 1
            results_list = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--result_path", action='store', type=str,
                        default="evaluation",
                        help='Where the json which should be evaluated is saved.')
    parser.add_argument('-d', "--directory", action='store', type=str, default="evaluation",
                        help='Dictionary where to evaluation results.')
    parser.add_argument("-p", "--plot", action='store_true', default=True, help='Whether to plot the results.')

    args = parser.parse_args()

    _main(args.result_path, args.directory, args.plot)
