import json

import pandas as pd
from tqdm import tqdm


def training_acc():
    latex = pd.DataFrame()

    for i, exp in tqdm(zip(range(6), [0, 1, 2, 4, 7, 8])):
        data = pd.read_csv("experiment_results/andreas_{}/01_dataset_creation_stats.csv".format(exp))

        acc_test = data["acc_test"]

        latex.loc["acc_test", "exp_{}".format(i)] = round(acc_test.mean() * 100, 1)

    with open("latex_acc_test.txt", "w") as file:
        latex.to_latex(file, header=True, index=True,
                       caption="The average test accuracy for the Random Forest classifier of the AL experiment.",
                       label="tabel:al_test_accuracy", column_format="lrrrrrrrrrrrrrr", position="tb", na_rep="-")


def auc_f1_eva():
    latex = pd.DataFrame()

    for i, exp in tqdm(zip(range(6), [0, 1, 2, 4, 7, 8])):
        data = pd.read_csv("experiment_results/andreas_{}/05_alipy_results.csv".format(exp))

        data.drop(
            columns=["strategy", "n_informative", "n_redundant", "n_repeated", "n_classes", "n_clusters_per_class",
                     "weights", "flip_y", "class_sep", "hypercube", "scale"], inplace=True)

        grouped_data = data.groupby(by=["dataset_id", "strategy_id"])

        strategies = {14: "ImitAL", 0: "Random", 2: "Uncertainty Max-Margin"}
        datasets = {2: "diabetes", 3: "fertility", 4: "german", 5: "haberman", 6: "heart", 8: "ionosphere",
                    10: "planning", 11: "australian", 14: "glass", 27: "zoo", 29: "flags", 16: "CIFAR-10", 12: "DWTC",
                    13: "EMNIST", 15: "OLIVETTI"}

        result = {}

        for idx, group in grouped_data.groups.items():
            g = data.loc[group]

            if idx[0] == 0: continue

            if idx[0] in datasets:
                ds_name = datasets[idx[0]]
            else:
                ds_name = idx[0]

            if ds_name not in result:
                result[ds_name] = {}

            if i == 0:
                latex.loc[ds_name, "samples"] = int(round(g["n_samples"].mean(), 0))
                latex.loc[ds_name, "features"] = int(round(g["n_features"].mean(), 0))

            latex.loc[ds_name, "exp_{}".format(i)] = round(g["f1_auc"].mean() * 100, 1)

            result[ds_name][strategies[idx[1]]] = {"duration": round(g["duration"].mean(), 2),
                                                   "f1_auc": round(g["f1_auc"].mean() * 100, 1),
                                                   "samples": round(g["n_samples"].mean(), 0),
                                                   "features": round(g["n_features"].mean(), 0)}

        with open("{}_results.json".format(i), "w") as file:
            json.dump(result, file)

    latex["samples"] = latex["samples"].astype(int)
    latex["features"] = latex["features"].astype(int)

    for column in latex.columns:
        latex.loc["mean", column] = round(latex[column].mean(), 1)

    with open("latex_results.txt", "w") as file:
        latex.to_latex(file, header=True, index=True, caption="The results of the experiments with the AL algorithm.",
                       label="tabel:al_evaluation_results", column_format="lrrrrrrrrrrrrrr", position="tb", na_rep="-")


if __name__ == "__main__":
    auc_f1_eva()
    training_acc()
