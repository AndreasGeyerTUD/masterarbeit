import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from generators.ml_datagen import generate
from tqdm import tqdm


def _round(matrix: np.array, res: int):
    return (np.round(matrix, len(str(res))) * res).astype(int)


def surface_plot(data, title: str, file_path: str, cmap: str = None):
    x = range(data.shape[0])
    y = range(data.shape[1])

    fig = plt.figure()

    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, data, cmap="RdYlGn_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.elev = 20

    plt.title(title)
    plt.savefig("{}_{}.pdf".format(file_path, cmap), format="pdf")
    # plt.show()
    plt.clf()


def heatmap_plot(data, title: str, file_path: str, cmap: str = None):
    fig, ax = plt.subplots()

    plt.imshow(data, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    ax.set_title(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig("{}_{}.pdf".format(file_path, cmap), format="pdf")
    # plt.show()
    plt.clf()


def _main(save_dir: str):
    for res in [10, 100, 1000, 10000]:
        cwd = save_dir + "/" + str(res)
        Path(cwd).mkdir(parents=True, exist_ok=True)

        results = np.zeros((res + 1, res + 1)).astype(int)
        label_1 = np.zeros((res + 1, res + 1)).astype(int)
        label_2 = np.zeros((res + 1, res + 1)).astype(int)

        for i in tqdm(range(res * 10)):
            # x, y = make_classification(n_samples=res * 10, n_features=2, n_informative=2, n_redundant=0,
            #                            n_clusters_per_class=1,
            #                            random_state=i)

            x, y, _ = generate(n_samples=res * 10, m_rel=2, m_irr=0, m_red=0, n_classes=2, n_clusters_per_class=1, shapes="mix", random_state=i, singlelabel=True)

            x = x.to_numpy()
            y = y.to_numpy()

            x_round = _round((x - x.min(0)) / x.ptp(0), res)

            # data = np.concatenate([x_round, y[:, None]], axis=1)

            for (x_1, x_2), y in zip(x_round, y):
                results[x_1][x_2] += 1

                if y == 0:
                    label_1[x_1][x_2] += 1
                if y == 1:
                    label_2[x_1][x_2] += 1

        pd = "Points Distribution"
        hm = "Heatmap"
        sp = "Surface-Plot"

        heatmap_plot(results, "{} - {}".format(pd, hm), "{}/heatmap".format(cwd), "gist_heat")
        heatmap_plot(results, "{} - {}".format(pd, hm), "{}/heatmap".format(cwd), "tab20c_r")
        heatmap_plot(label_1, "{} - {} - Label 0".format(pd, hm), "{}/heatmap_label_0".format(cwd), "gist_heat")
        heatmap_plot(label_1, "{} - {} - Label 0".format(pd, hm), "{}/heatmap_label_0".format(cwd), "tab20c_r")
        heatmap_plot(label_2, "{} - {} - Label 1".format(pd, hm), "{}/heatmap_label_1".format(cwd), "gist_heat")
        heatmap_plot(label_2, "{} - {} - Label 1".format(pd, hm), "{}/heatmap_label_1".format(cwd), "tab20c_r")

        surface_plot(results, "{} - {}".format(sp, pd), "{}/surface_plot".format(cwd), "RdYlGn_r")
        surface_plot(label_1, "{} - {} - Label 0".format(sp, pd), "{}/surface_plot_label_0".format(cwd), "RdYlGn_r")
        surface_plot(label_2, "{} - {} - Label 1".format(sp, pd), "{}/surface_plot_label_1".format(cwd), "RdYlGn_r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="points_distribution_ml_datagen",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    _main(args.directory)
