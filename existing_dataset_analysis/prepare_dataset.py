import pathlib

import pandas as pd
import yaml


with open("dataset_parameters.yaml", 'r') as params_file:
    parameter_dict = yaml.safe_load(params_file)


def read_and_parse_dataset(dataset_path: str, save_path: str):
    dataset_path = pathlib.Path(dataset_path)
    dataset_name = dataset_path.name.split(".")[0]
    parsing_args = parameter_dict[dataset_name]

    with open(dataset_path, "r") as f:
        df = pd.read_csv(f, sep=",")

    for column, dtype in df.dtypes.items():
        if dtype not in ["int64", "float64"]:
            if dtype.name != "category":
                df[column] = df[column].astype('category')
            df[column] = df[column].cat.codes

    label_column = parsing_args["target"]
    if label_column is not None:
        if isinstance(label_column, list):
            labels = df[label_column]
        else:
            labels = df[label_column].to_frame()
        df.drop(label_column, axis=1, inplace=True)
    else:
        labels = None

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(pathlib.PurePath.joinpath(save_path, parsing_args["save_name"] + "_dataset.csv"), index=False)
        if labels is not None:
            labels.to_csv(pathlib.PurePath.joinpath(save_path, parsing_args["save_name"] + "_labels.csv"), index=False)

    return df, labels


if __name__ == "__main__":
    read_and_parse_dataset("uncleaned_datasets/zoo.csv", "cleaned_datasets_old")
