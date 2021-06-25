import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import make_classification
from tqdm import tqdm


def _round(matrix: np.array, res: int):
    return (np.round(matrix, len(str(res))) * res).astype(int)


def heatmap_plot(data, file_path: str):
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
            x, y = make_classification(n_samples=res * 10, n_features=3, n_informative=3, n_redundant=0,
                                       n_clusters_per_class=1, random_state=i)

            data = _round((x - x.min(0)) / x.ptp(0), res)

            if whole_data is None:
                whole_data = data
            else:
                whole_data = np.concatenate([whole_data, data], axis=0)

        heatmap_plot(whole_data, "{}/density_plot".format(cwd))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="points_distribution_3d",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    _main(args.directory)
