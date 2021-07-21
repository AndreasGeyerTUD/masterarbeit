import pathlib

import pandas as pd
import yaml


with open("parameters.yaml", 'r') as params_file:
    parameter_dict = yaml.safe_load(params_file)


def read_and_parse_dataset(dataset_path: str, save_path: str):
    dataset_path = pathlib.Path(dataset_path)
    dataset_name = dataset_path.name.split(".")[0]
    parsing_args = parameter_dict[dataset_name]

    with open(dataset_path, "r") as f:
        df = pd.read_csv(f, sep=parsing_args[0], header=parsing_args[1], index_col=parsing_args[2])

    if dataset_name == "PLANNING_plrx":
        df.drop(13, axis=1, inplace=True)
    if dataset_name == "BREAST":
        df.drop("Unnamed: 32", axis=1, inplace=True)
    if dataset_name == "adult":
        df["Age"].astype("int32")
    if dataset_name == "zoo":
        df.drop("animal_name", axis=1, inplace=True)
    # if dataset_name == "parkinsons":
    #     df.drop("name", axis=1, inplace=True)
    if dataset_name == "flag":
        df.drop(0, axis=1, inplace=True)
    if dataset_name == "ipl":
        df.drop(["BBI", "y"], axis=1, inplace=True)

    for column, dtype in df.dtypes.items():
        if dtype not in ["int64", "float64"]:
            if dtype.name != "category":
                df[column] = df[column].astype('category')
            df[column] = df[column].cat.codes

    for column in df.columns:
        if (df[column] < 0).any():
            df[column] = df[column] + abs(df[column].min())

    label_column = parsing_args[3]
    if label_column is not None:
        labels = df[label_column]
        df.drop(label_column, axis=1, inplace=True)
    else:
        labels = None

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(pathlib.PurePath.joinpath(save_path, dataset_name + "_dataset.csv"), index=False)
        if labels is not None:
            labels.to_csv(pathlib.PurePath.joinpath(save_path, dataset_name + "_labels.csv"), index=False)

    return df, labels


if __name__ == "__main__":
    read_and_parse_dataset("uncleaned_datasets/zoo.csv",
                           "cleaned_datasets")
