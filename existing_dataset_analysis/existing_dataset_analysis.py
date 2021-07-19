from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits, load_linnerud, load_wine, \
    load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_scatter_matrix(data: pd.DataFrame, name: str):
    sns.set_theme(style="white")

    splits = split_columns_for_plot(data.columns)

    for i in range(len(splits)):
        for j in range(len(splits)):
            g = sns.PairGrid(data, diag_sharey=False, x_vars=splits[i], y_vars=splits[j])
            if i == j:
                g.map_upper(sns.scatterplot)
                g.map_lower(sns.kdeplot)
                g.map_diag(sns.kdeplot)
            elif i < j:
                g.map(sns.kdeplot)
            elif i > j:
                g.map(sns.scatterplot)

            g.fig.subplots_adjust(top=0.95)
            g.fig.suptitle(name)

            g.savefig("dataset_plots/{}_{}_{}.pdf".format(name, i, j), format="pdf")
            plt.close()


def sklearn_datasets():
    datasets = [load_boston, load_iris, load_diabetes, load_wine, load_breast_cancer]

    see_what_is_possible = [load_digits, load_linnerud]

    for dataset in datasets:
        ds = dataset()
        if "filename" in ds:
            name = ds.filename.split("\\")[-1].split("/")[-1].split(".")[0]
        else:
            name = str(dataset).split(" ")[1].split("_")[1]

        x = pd.DataFrame(ds.data, columns=ds.feature_names)

        plot_scatter_matrix(x, name)


def seaborn_datasets():
    datasets = ["car_crashes", "penguins", "exercise", "mpg"]

    for dataset in datasets:
        data = sns.load_dataset(dataset)
        name = dataset

        if dataset == "car_crashes":
            data.drop("abbrev", axis=1, inplace=True)
        elif dataset == "exercise":
            data.drop("Unnamed: 0", axis=1, inplace=True)
            data["time"] = data["time"].apply(str_split).astype(int)
        elif dataset == "mpg":
            data.drop("name", axis=1, inplace=True)

        for column in data.columns:
            if data[column].dtype == object:
                data[column] = data[column].astype('category')
            if data[column].dtype.name == "category":
                data[column] = data[column].cat.codes

        print()

        plot_scatter_matrix(data, name)


def str_split(string: str) -> str:
    return string.split(" ")[0]


if __name__ == "__main__":
    # sklearn_datasets()
    seaborn_datasets()
