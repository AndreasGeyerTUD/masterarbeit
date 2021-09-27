import json

import pandas as pd

if __name__ == "__main__":
    for i in range(6):
        data = pd.read_csv("0{}_05_alipy_results.csv".format(i))

        data.drop(
            columns=["strategy", "n_informative", "n_redundant", "n_repeated", "n_classes", "n_clusters_per_class",
                     "weights", "flip_y", "class_sep", "hypercube", "scale"], inplace=True)

        grouped_data = data.groupby(by=["dataset_id", "strategy_id"])

        strategies = {14: "ImitAL", 0: "Random", 2: "Uncertainty Max-Margin"}
        datasets = {2: "DIABETES", 3: "FERTILITY", 4: "GERMAN", 5: "HABERMAN", 6: "HEART", 8: "IONOSPHERE",
                    10: "PLANNING", 11: "australian", 14: "glass", 27: "zoo", 29: "flag"}

        result = {}

        for idx, group in grouped_data.groups.items():
            g = data.loc[group]

            result[datasets[idx[0]]] = {}
            result[datasets[idx[0]]][strategies[idx[1]]] = {"duration": round(g["duration"].mean(), 2),
                                                            "f1_auc": round(g["f1_auc"].mean(), 3),
                                                            "samples": round(g["n_samples"].mean(), 0),
                                                            "features": round(g["n_features"].mean(), 0)}

        with open("{}_results.json".format(i), "w") as file:
            json.dump(result, file)
