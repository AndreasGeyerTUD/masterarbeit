import pathlib

import pandas as pd

parsing_dict = {
    "dwtc": [",", 0, 0, "CLASS"],
    "PLANNING_plrx": ["\t", None, None, 12],
    "GERMAN_credit_data.data": ["\s+", None, None, 24],
    "FERTILITY": [",", 0, None, "Diagnosis"],
    "PIMA-indians-diabetes": [",", 0, None, "Outcome"],
    "australian.dat": [" ", None, None, 14],
    "ILPD_Indian Liver Patient Dataset (ILPD)": [",", 0, None, "label"],
    "IONOSPHERE_ionosphere.data": [",", None, None, 34],
    "HEART": [",", 0, None, "target"],
    "HABERMAN": [",", None, None, 3],
    "BREAST": [",", 0, None, "diagnosis"],
    "DIABETES": [",", 0, None, "Outcome"],
    "abalone": [",", 0, None, "label"],
    "adult": [",", 0, None, "Target"],
    #  "wine": [",", 0, None, "Type"],
    "wine": [",", 0, None, "Class"],
    "glass": [",", 0, None, "Type"],
    "parkinsons": [",", 0, None, "status"],
    "zoo": [",", 0, None, "class_type"],
    "flag": [",", None, None, 6],
    #  "hepatitis": [",", None, None, 0], # missing values!
}


def read_and_parse_dataset(dataset_path: str, save_path: str):
    dataset_path = pathlib.Path(dataset_path)
    dataset_name = dataset_path.name.split(".")[0]
    parsing_args = parsing_dict[dataset_name]

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
    if dataset_name == "parkinsons":
        df.drop("name", axis=1, inplace=True)
    if dataset_name == "flag":
        df.drop(0, axis=1, inplace=True)

    label_column = parsing_args[3]
    if label_column is not None:
        labels = df[label_column]
        df.drop(label_column, axis=1, inplace=True)
    else:
        labels = None

    for column, dtype in df.dtypes.items():
        if dtype not in ["int64", "float64"]:
            if len(df[column].unique()) > 2:
                df = pd.concat([pd.get_dummies(df[column], prefix=column), df.drop(column, axis=1)], axis=1)
            else:
                df.loc[:, column] = df.loc[:, column].astype("category").cat.codes

    if save_path:
        save_path = pathlib.Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(pathlib.PurePath.joinpath(save_path, dataset_name + "_dataset.csv"), index=False)
        if labels is not None:
            labels.to_csv(pathlib.PurePath.joinpath(save_path, dataset_name + "_labels.csv"), index=False)

    return df, labels


if __name__ == "__main__":
    read_and_parse_dataset("../../existing_dataset_analysis/uci_datasets/zoo.csv",
                           "../../existing_dataset_analysis/uci_cleaned")
