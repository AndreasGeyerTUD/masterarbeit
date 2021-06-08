from sklearn.datasets import make_classification
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def _round(matrix: np.array):
    return (np.round(matrix, 4) * 1000).astype(int)


def surface_plot(data):
    x = range(data.shape[0])
    y = range(data.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, data, cmap="gist_earth")

    plt.title("Surface-Plot - Points Distribution")
    plt.savefig("points_surface_plot.pdf", format="pdf")
    plt.show()


def heatmap_plot(data, title: str, file_path: str, cmap: str = None):
    ax = sns.heatmap(data, cmap=cmap, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.savefig("{}.pdf".format(file_path), format="pdf")
    plt.show()


def _main():
    results = np.zeros((1001, 1001)).astype(int)

    for i in tqdm(range(10000)):
        x, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1,
                                   random_state=i)

        x_round = _round((x - x.min(0)) / x.ptp(0))

        # TODO: result is not equal... randomly picked 2 values -> more than 20 difference (less)... need to check this!
        # results[x_round[:, 0], x_round[:, 1]] += 1

        for x_1, x_2 in x_round:
            results[x_1][x_2] += 1

    heatmap_plot(results, "Heatmap - Points Distribution", "points_distribution/points_heatmap")
    heatmap_plot(results, "Heatmap - Points Distribution", "points_distribution/points_heatmap_tab20c_r", "tab20c_r")

    surface_plot(results)


if __name__ == "__main__":
    _main()
