from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits, load_linnerud, load_wine, \
    load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

datasets = [load_boston, load_iris, load_diabetes, load_wine, load_breast_cancer]

see_what_is_possible = [load_digits, load_linnerud]


def sklearn_datasets():
    for dataset in datasets:
        ds = dataset()
        if "filename" in ds:
            name = ds.filename.split("\\")[-1].split(".")[0]
        else:
            name = str(dataset).split(" ")[1].split("_")[1]

        x = pd.DataFrame(ds.data, columns=ds.feature_names)

        pd.plotting.scatter_matrix(x, diagonal="hist", figsize=(20, 20), alpha=1)

        # x.plot(y=ds.target, kind="scatter", subplots=True)
        plt.savefig("dataset_plots/{}.pdf".format(name), format="pdf")
        plt.show()

        print()


if __name__ == "__main__":
    sklearn_datasets()