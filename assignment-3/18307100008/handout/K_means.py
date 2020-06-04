import numpy as np
from sklearn.metrics import pairwise_distances_argmin as assign


class KMeans:

    def __init__(self, k):
        self.k = k

    def fit(self, data):
        print("Start K-Means Initialize...")
        centroids = data[np.random.choice(data.shape[0], size=self.k, replace=False), :]
        last_cluster = np.zeros(data.shape[0])
        counter = 0
        while True:
            # print("Start iteration:", counter)
            cluster = assign(data, centroids)
            if all(cluster == last_cluster):
                break
            centroids = np.array([np.mean(data[np.where(cluster == i)], axis=0) for i in range(self.k)])
            last_cluster = cluster
            counter += 1
        print("End K-Means iteration.")
        self.cluster = cluster
        self.centroids = centroids
        return self.cluster
