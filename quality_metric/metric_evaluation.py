import argparse

import numpy as np
import json


def singlelabel(data: dict) -> dict:
    helper = {}

    for date in data.values():
        for entry, value in date.items():
            if entry not in helper:
                helper[entry] = {}
            for key, val in value.items():
                if val is None: continue
                if key not in helper[entry]:
                    helper[entry][key] = [val]
                else:
                    helper[entry][key].append(val)

    evaluation = {}

    for key, value in helper.items():
        if key == "dataset_information": continue
        if key not in evaluation:
            evaluation[key] = {}
        for k, v in value.items():
            if k == "best_params" or k == "opt_score": continue
            evaluation[key][k] = np.mean(v)

    return evaluation


def multilabel(data: dict) -> dict:
    helper = {}

    for date in data.values():
        for method, value in date.items():
            if method not in helper:
                helper[method] = {}
            if method == "ml_knn":
                for met, val in value.items():
                    if met not in helper[method]:
                        helper[method][met] = {}
                    for k, v in val.items():
                        if v is None: continue
                        if k not in helper[method]:
                            helper[method][met][k] = [v]
                        else:
                            helper[method][met][k].append(v)
            else:
                for cla, val in value.items():
                    if val is None: continue
                    if cla not in helper[method]:
                        helper[method][cla] = [val]
                    else:
                        helper[method][cla].append(val)

    evaluation = {}


    return data


def _main(result_path: str, save_path: str):
    with open(result_path, "r") as file:
        data = json.load(file)

    type = data["type"]

    data.pop("type", None)

    if type == "singlelabel":
        evaluation = singlelabel(data)
    elif type == "multilabel":
        evaluation = multilabel(data)
    else:
        evaluation = {}

    evaluation["dataset_information"] = data["0"]["dataset_information"]
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

    args = parser.parse_args()

    _main(args.result_path, args.directory)
