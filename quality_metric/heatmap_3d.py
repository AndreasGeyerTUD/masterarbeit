import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import make_classification, make_blobs
from tqdm import tqdm

from generators.ml_datagen import generate


def _round(matrix: np.array, res: int):
    return (np.round(matrix, len(str(res))) * res).astype(int)


def heatmap_plot_2d(data: np.array, res: int, title: str, file_path: str, cmap: str = None):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    for v_1, n_1, v_2, n_2 in [(x, "x", y, "y"), (y, "y", z, "z"), (x, "x", z, "z")]:
        cur_data = np.zeros((res + 1, res + 1)).astype(int)
        for i, j in zip(v_1, v_2):
            cur_data[i][j] += 1

        fig, ax = plt.subplots()

        cur_title = title + "_{}{}".format(n_1, n_2)

        plt.imshow(cur_data, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        ax.set_title(cur_title)
        plt.xlabel(n_1)
        plt.ylabel(n_2)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig("{}/{}_{}.pdf".format(file_path, cur_title, cmap), format="pdf")
        # plt.show()
        plt.close()


def heatmap_plot_3d(data, file_path: str):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    xyz = np.vstack([x, y, z])
    density = stats.gaussian_kde(xyz)(xyz)

    points_size = ((density - density.min(0)) / density.ptp(0)) * 30

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density, s=points_size, cmap="RdYlGn_r", alpha=.02)

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


def _main(save_dir: str):
    for res in [10, 100, 1000]:
        cwd = save_dir + "/" + str(res)
        Path(cwd).mkdir(parents=True, exist_ok=True)

        whole_data = None

        for i in tqdm(range(res * 10)):
            # x, y = make_classification(n_samples=res * 10, n_features=3, n_informative=3, n_redundant=0,
            #                            n_clusters_per_class=1, random_state=i)

            # x, y, _ = generate(n_samples=res * 10, m_rel=3, m_irr=0, m_red=0, n_classes=1, n_clusters_per_class=1,
            #                    shapes="mix", random_state=i, singlelabel=True)
            #
            # x = x.to_numpy()
            # y = y.to_numpy()

            x, y = make_blobs(n_samples=res * 10, centers=1, n_features=3, random_state=i)

            data = _round((x - x.min(0)) / x.ptp(0), res)

            if whole_data is None:
                whole_data = data
            else:
                whole_data = np.concatenate([whole_data, data], axis=0)

        if res <= 100: heatmap_plot_3d(whole_data, "{}/density_plot".format(cwd))
        heatmap_plot_2d(whole_data, res, "heatmap", cwd, "gist_heat")
        heatmap_plot_2d(whole_data, res, "heatmap", cwd, "tab20c_r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="points_distribution_3d_make_blobs",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    _main(args.directory)
