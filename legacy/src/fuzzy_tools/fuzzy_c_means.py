## Fuzzy c means from the internet
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt


def c_means(X, nodes, plot=False):
    fcm = FCM(n_clusters=nodes)
    fcm.fit(X)
    fcm_centers = fcm.centers
    if plot:
        fcm_labels = fcm.predict(X)
        # plot result
        f, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.1)
        axes[1].scatter(X[:, 0], X[:, 1], c=fcm_labels, alpha=0.1)
        axes[1].scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=500, c="w")
        axes[1].set(xlim=(0, 800), ylim=(0, 600))
        plt.show()
    return fcm_centers
