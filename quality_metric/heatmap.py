import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm


def _round(matrix: np.array):
    return (np.round(matrix, 4) * 1000).astype(int)


def surface_plot(data, title: str, file_path: str, cmap: str = None):
    x = range(data.shape[0])
    y = range(data.shape[1])

    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, data, cmap="RdYlGn_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.elev = 20

    plt.title(title)
    plt.savefig("{}_{}.pdf".format(file_path, cmap), format="pdf")
    plt.show()


def heatmap_plot(data, title: str, file_path: str, cmap: str = None):
    fig, ax = plt.subplots()

    plt.imshow(data, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    ax.set_title(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig("{}_{}.pdf".format(file_path, cmap), format="pdf")
    plt.show()


def _main(save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = np.zeros((1001, 1001)).astype(int)
    label_1 = np.zeros((1001, 1001)).astype(int)
    label_2 = np.zeros((1001, 1001)).astype(int)

    for i in tqdm(range(10000)):
        x, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=i)

        x_round = _round((x - x.min(0)) / x.ptp(0))

        data = np.concatenate([x_round, y[:, None]], axis=1)

        for x_1, x_2, y in data:
            results[x_1][x_2] += 1

            if y == 0:
                label_1[x_1][x_2] += 1
            if y == 1:
                label_2[x_1][x_2] += 1

    pd = "Points Distribution"
    hm = "Heatmap"
    sp = "Surface-Plot"

    heatmap_plot(results, "{} - {}".format(pd, hm), "{}/heatmap".format(save_dir), "gist_heat")
    heatmap_plot(results, "{} - {}".format(pd, hm), "{}/heatmap".format(save_dir), "tab20c_r")
    heatmap_plot(label_1, "{} - {} - Label 0".format(pd, hm), "{}/heatmap_label_0".format(save_dir), "gist_heat")
    heatmap_plot(label_1, "{} - {} - Label 0".format(pd, hm), "{}/heatmap_label_0".format(save_dir), "tab20c_r")
    heatmap_plot(label_2, "{} - {} - Label 1".format(pd, hm), "{}/heatmap_label_1".format(save_dir), "gist_heat")
    heatmap_plot(label_2, "{} - {} - Label 1".format(pd, hm), "{}/heatmap_label_1".format(save_dir), "tab20c_r")

    surface_plot(results, "{} - {}".format(sp, pd), "{}/surface_plot".format(save_dir), "RdYlGn_r")
    surface_plot(label_1, "{} - {} - Label 0".format(sp, pd), "{}/surface_plot_label_0".format(save_dir), "RdYlGn_r")
    surface_plot(label_2, "{} - {} - Label 1".format(sp, pd), "{}/surface_plot_label_1".format(save_dir), "RdYlGn_r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="points_distribution",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    _main(args.directory)
