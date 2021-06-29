import argparse

import numpy as np
import json


def _main(result_path: str, save_path:str):
    with open(result_path, "r") as file:
        data = json.load(file)

    helper = {}

    for _, date in data.items():
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

    evaluation["dataset_information"] = data["0"]["dataset_information"]
    evaluation["number_of_datasets"] = len(data)

    with open("{}/evaluation.json".format(save_path), "w") as file:
        json.dump(evaluation, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--result_path", action='store', type=str, default="singlelabel_test_dataset/results.json",
                        help='Where the json which should be evaluated is saved.')
    parser.add_argument('-d', "--directory", action='store', type=str, default="singlelabel_test_dataset",
                        help='Dictionary where to evaluation results.')

    args = parser.parse_args()

    _main(args.result_path, args.directory)
