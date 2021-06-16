from mayavi import mlab
import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from tqdm import tqdm
from scipy import stats


def _round(matrix: np.array, res: int):
    return (np.round(matrix, len(str(res))) * res).astype(int)


def heatmap_plot(data, title: str, file_path: str, res: int):
    xyz = np.vstack([data[:, 0], data[:, 1], data[:, 2]])
    kde = stats.gaussian_kde(xyz)
    density = kde(xyz)

    figure = mlab.figure('DensityPlot', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.title(title)
    pts = mlab.points3d(data[:, 0], data[:, 1], data[:, 2], density)
    mlab.axes(color=(0, 0, 0), ranges=[0, res, 0, res, 0, res])

    for i in range(1, 4):
        for j in range(1, 4):
            mlab.view(i * 45, j * 45, 40)
            for file_type in ["png", "jpg", "tiff", "bmp"]:
                mlab.savefig("{}_{}_{}.{}".format(file_path, i * 45, j * 45, file_type), figure=figure)

    mlab.close()


def _main(save_dir: str):
    for res in [10, 100, 1000]:
        cwd = save_dir + "/" + str(res)
        Path(cwd).mkdir(parents=True, exist_ok=True)

        whole_data = None

        for i in tqdm(range(res * 10)):
            x, y = make_classification(n_samples=res * 10, n_features=3, n_informative=3, n_redundant=0,
                                       n_clusters_per_class=1, random_state=i)

            x_round = _round((x - x.min(0)) / x.ptp(0), res)

            data = x_round

            if whole_data is None:
                whole_data = data
            else:
                whole_data = np.concatenate([whole_data, data], axis=0)

        heatmap_plot(whole_data, "Points Distribution - DensityPlot", "{}/density_plot".format(cwd), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--directory", action='store', type=str, default="points_distribution_3d",
                        help='Dictionary where to save images and results.')

    args = parser.parse_args()

    _main(args.directory)
