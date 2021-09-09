from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston, load_iris, load_diabetes, load_wine, \
    load_breast_cancer
from tqdm import tqdm

from prepare_dataset import read_and_parse_dataset


def split_columns_for_plot(columns: list) -> list[list]:
    if len(columns) > 10:
        splitted_columns = []
        split = int(len(columns) / 2)
        first = columns[split:]
        second = columns[:split]
        if split > 10:
            for spl in split_columns_for_plot(first):
                splitted_columns.append(spl)
            for spl in split_columns_for_plot(second):
                splitted_columns.append(spl)
        else:
            splitted_columns.append(first)
            splitted_columns.append(second)

        return splitted_columns
    else:
        return [columns]


def plot_scatter_matrix(data: pd.DataFrame, name: str, labels: pd.DataFrame = None, origin: str = None,
                        force: bool = False):
    sns.set_theme(style="white")

    if len(data.columns) > 40 and not force:
        raise TimeoutError("You tried to plot a dataset with more than 40 dimensions. If you intend to do this, then "
                           "set 'force=True'!")

    splits = split_columns_for_plot(data.columns)

    if labels is None:
        labels_column = None
    elif len(labels[labels.columns[0]].unique()) > 10:
        print("This dataset contains more than 10 unique labels. This seems to be a regression dataset and labels are "
              "therefore not highlighted.")
        labels_column = None
    else:
        labels_column = labels.columns[0]
        data[labels_column] = labels[labels_column]

    for i in range(len(splits)):
        for j in range(len(splits)):
            g = sns.PairGrid(data, hue=labels_column, diag_sharey=False, x_vars=splits[i], y_vars=splits[j])
            if i == j:
                g.map_upper(sns.scatterplot, linewidth=0)
                g.map_lower(sns.kdeplot)
                g.map_diag(sns.kdeplot)
            elif i < j:
                g.map(sns.kdeplot)
            elif i > j:
                g.map(sns.scatterplot, linewidth=0)

            g.fig.subplots_adjust(top=0.95)
            g.fig.suptitle(name)

            if labels_column is not None:
                g.add_legend()

            if len(splits) > 1:
                save_name = "{}_{}_{}".format(name, i, j)
            else:
                save_name = "{}".format(name)

            if origin is not None:
                save_name = "{}_{}".format(origin, save_name)

            g.savefig("dataset_plots/{}.pdf".format(save_name), format="pdf")
            # plt.show()
            plt.close()


def sklearn_datasets():
    datasets = [load_iris, load_boston, load_diabetes, load_wine, load_breast_cancer]

    for dataset in datasets:
        ds = dataset()
        if "filename" in ds:
            name = ds.filename.split("\\")[-1].split("/")[-1].split(".")[0]
        else:
            name = str(dataset).split(" ")[1].split("_")[1]

        df = pd.DataFrame(ds.data, columns=ds.feature_names)
        if "target" in ds:
            labels = pd.DataFrame(ds["target"], columns=["LABELS"])
        else:
            labels = None

        plot_scatter_matrix(df, name, labels, "sklearn")


def seaborn_datasets():
    datasets = ["car_crashes", "penguins", "exercise", "mpg"]

    for dataset in datasets:
        data = sns.load_dataset(dataset)
        name = dataset

        labels = None

        if dataset == "car_crashes":
            data.drop("abbrev", axis=1, inplace=True)
        elif dataset == "exercise":
            data.drop("Unnamed: 0", axis=1, inplace=True)
            data["time"] = data["time"].apply(str_split).astype(int)
        elif dataset == "mpg":
            data.drop("name", axis=1, inplace=True)
        elif dataset == "penguins":
            labels = data["species"].to_frame()
            data.drop("species", axis=1, inplace=True)

        for column, dtype in data.dtypes.items():
            if dtype not in ["int64", "float64"]:
                if dtype.name != "category":
                    data[column] = data[column].astype('category')
                data[column] = data[column].cat.codes

        plot_scatter_matrix(data, name, labels, "seaborn")


def str_split(string: str) -> str:
    return string.split(" ")[0]


def other_datasets():
    path = Path("uncleaned_datasets/")
    files = list(path.glob("*"))

    for file in tqdm(files):
        name = file.name.split(".")[0]

        if not Path("cleaned_datasets/{}_dataset.csv".format(name)).exists():
            dataset, labels = read_and_parse_dataset(str(file), "cleaned_datasets")
        else:
            dataset = pd.read_csv("cleaned_datasets/{}_dataset.csv".format(name), sep=",", header=0)
            if Path("cleaned_datasets/{}_labels.csv".format(name)).exists():
                labels = pd.read_csv("cleaned_datasets/{}_labels.csv".format(name), sep=",", header=0)
            else:
                labels = None

        try:
            plot_scatter_matrix(dataset, name, labels, "other")
        except Exception as e:
            print(e, name)


if __name__ == "__main__":
    # sklearn_datasets()
    # seaborn_datasets()
    other_datasets()

problems = ["bank", "blood", "credit", "elections", "facebook"]
large = ["elections", "gold", "house", "mice"]
