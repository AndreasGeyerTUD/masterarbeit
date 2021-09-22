from sklearn.datasets import make_classification, make_moons, make_blobs, make_circles, make_regression
import matplotlib.pyplot as plt
from generators.ml_datagen import generate


# plt.plot(x[:, 0], x[:, 1], "o")
# plt.show()

# x, y, _ = generate(1000, "moons", 2, 0, 0, 1, 1, singlelabel=True, random_state=4)


def plot_sl(dataset, labels, name: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = dataset[:,0]
    y = dataset[:,1]
    # plt.xlim(-3, 3)
    # plt.ylim(-3, 3)
    ax.set_aspect('equal', adjustable='box')
    # plt.xticks([-2, 0, 2])
    # plt.yticks([-2, 0, 2])
    plt.scatter(x, y, c=labels)
    plt.title(name)
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

x, y = make_regression(n_samples=1000, n_features=2, n_informative=2, random_state=3)

plot_sl(x,y, "make_regression")