import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_blobs, make_circles, make_regression
import matplotlib.pyplot as plt
from generators.ml_datagen import generate
import seaborn as sns

import multiprocessing as mp


# plt.plot(x[:, 0], x[:, 1], "o")
# plt.show()

# x, y, _ = generate(1000, "moons", 2, 0, 0, 1, 1, singlelabel=True, random_state=4)


def plot_sl(dataset, labels, name: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = dataset[:, 0]
    y = dataset[:, 1]
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    # plt.xticks([-2, 0, 2])
    # plt.yticks([-2, 0, 2])
    plt.scatter(x, y, c=labels)
    # plt.title(name)
    plt.savefig(name)
    plt.show()


# plot_sl(x, y, "test.png")

# x, y, _ = generate(5000, "moons", 3, 0, 0, 1, 1, singlelabel=True, random_state=4)


def heatmap_plot_3d(data, file_path: str):
    data = data.to_numpy()
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # plt.draw()
    # plt.show()
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    for i in range(1, 16):
        if i % 2 == 0:
            continue
        for j in range(1, 16):
            if j % 4 == 0:
                continue
            ax.view_init(i * 22.5, j * 22.5)
            plt.draw()
            plt.savefig("{}_{}_{}.png".format(file_path, i * 22.5, j * 22.5), format="png")

    plt.close()


# plot_sl(x, y, "test_2.png")
# heatmap_plot_3d(x, "test/test")


# x, y = make_classification(n_samples=1000, n_classes=2, n_features=2, n_informative=2, n_redundant=0,
#                            n_clusters_per_class=1, random_state=3)
#
# plot_sl(x,y, "make_classification")
#
# x, y = make_circles(n_samples=1000, random_state=3)
#
# plot_sl(x,y, "make_circles")
#
# x, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=3)
#
# plot_sl(x,y, "make_blobs")
#
# x, y = make_moons(n_samples=1000, random_state=3)
#
# plot_sl(x,y, "make_moons")

# x, y = make_regression(n_samples=1000, n_features=2, n_informative=2, random_state=3)
#
# plot_sl(x,y, "make_regression")

# x, y, _ = generate(n_samples=5000, shapes=[("cubes", 0.34), ("spheres", 0.33), ("moons", 0.33)], m_rel=2, m_irr=0,
#                    max_r=0.4, min_r=0.2, m_red=0, n_classes=3, n_clusters_per_class=1, random_state=9)
#
# x = x.to_numpy()
# y = y.to_numpy()
#
# plot_sl(x, y, "csm_shapes")

# x, y, _ = generate(n_samples=5000, shapes=[("cubes", 0.34), ("spheres", 0.33), ("moons", 0.33)], m_rel=2, m_irr=0,
#                    max_r=0.4, min_r=0.2, m_red=0, n_classes=3, n_clusters_per_class=1, random_state=9, random_points=0.2)
#
# x = x.to_numpy()
# y = y.to_numpy()
#
# plot_sl(x, y, "csm_shapes_point_noise")

# x, _, y = generate(n_samples=5000, shapes=[("cubes", 0.34), ("spheres", 0.33), ("moons", 0.33)], m_rel=2, m_irr=0,
#                    max_r=0.4, min_r=0.2, m_red=0, n_classes=3, n_clusters_per_class=1, random_state=9, noise_levels=[0.2])
#
# x = x.to_numpy()
# y = y[0].to_numpy()
#
# plot_sl(x, y, "csm_shapes_noise_level")

# x, y, _ = generate(n_samples=5000, shapes=[("cubes", 0.34), ("spheres", 0.33), ("moons", 0.33)], m_rel=2, m_irr=0,
#                    max_r=0.4, min_r=0.2, m_red=0, n_classes=3, n_clusters_per_class=1, random_state=9, mov_vectors="random")
#
# x = x.to_numpy()
# y = y.to_numpy()
#
# plot_sl(x, y, "csm_shapes_transformation")


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

    splits = split_columns_for_plot(data.columns.to_list())

    if labels is None:
        labels_column = None
    elif len(labels[labels.columns[0]].unique()) > 10:
        print("This dataset contains more than 10 unique labels. This seems to be a regression dataset and labels are "
              "therefore not highlighted.")
        labels_column = None
    else:
        labels_column = labels.columns[0]
        data[labels_column] = labels[labels_column]

    data = data.sample(frac=1).reset_index(drop=True)

    for i in range(len(splits)):
        for j in range(len(splits)):
            g = sns.PairGrid(data, hue=labels_column, palette="tab10", diag_sharey=False, x_vars=splits[i], y_vars=splits[j])
            if i == j:
                g.map_upper(sns.scatterplot, linewidth=0)
                g.map_lower(sns.kdeplot, warn_singular=False)
                g.map_diag(sns.kdeplot, warn_singular=False)
            elif i < j:
                g.map(sns.kdeplot, warn_singular=False)
            elif i > j:
                g.map(sns.scatterplot, linewidth=0)

            # g.fig.subplots_adjust(top=0.95)
            # g.fig.suptitle(name)

            if labels_column is not None:
                g.add_legend()

            if len(splits) > 1:
                save_name = "{}_{}_{}".format(name, i, j)
            else:
                save_name = "{}".format(name)

            if origin is not None:
                save_name = "{}_{}".format(origin, save_name)

            g.savefig("plots/{}.png".format(save_name), format="png")
            # plt.show()
            plt.close()


def do(i: int):
    x, y, _ = generate(n_samples=1000, shapes="mix", m_rel=5, m_irr=0, m_red=0, n_classes=3, n_clusters_per_class=1,
                       random_state=i, mov_vectors="random")

    plot_scatter_matrix(x, str(i), y, "csm-datagen")

    x, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, n_classes=3,
                               random_state=i)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y, columns=["labels"])

    plot_scatter_matrix(x, str(i), y, "make_classification")

    x, y = make_blobs(n_samples=1000, n_features=5, centers=3, random_state=i)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y, columns=["labels"])

    plot_scatter_matrix(x, str(i), y, "make_blobs")


if __name__ == "__main__":
    pool = mp.Pool(10)
    pool.map(do, range(10))
