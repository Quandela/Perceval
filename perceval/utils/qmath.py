# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.spatial.distance import cdist


def exponentiation_by_squaring(base, power: int):
    """Calculate the result of base^power i.e. base**power using exponentiation by squaring (or square-and-multiply)

    Args:
        :param base: the element to exponentiate
        :param power: *strictly positive* integer power
        :param result: the initialisation of the result
    """
    if power < 1:
        raise ValueError("Power value must be strictly positive")

    if isinstance(base, int):
        temp_base = base
        result = base
    else:
        temp_base = base.__copy__()
        result = base.__copy__()

    power -= 1

    while True:
        # If power is odd
        if power % 2 == 1:
            result = result * temp_base

        # Divide the power by 2
        power = power // 2
        if power == 0:
            break
        # Multiply base to itself
        temp_base = temp_base * temp_base

    return result


def kmeans(features: np.ndarray, n_clusters: int, n_init: int = 10) -> np.ndarray:
    """
    Manual KMeans implementation. Clusterizes the system in k subsets.

    :param features: Data points for clustering.
    :param n_clusters: Number of clusters.
    :param n_init: Number of times the k-means algorithm will be run with different centroid seeds.
    :return: Cluster labels.
    :rtype: np.ndarray
    """
    best_labels = None
    best_inertia = float('inf')
    MAX_ITERATIONS = 300
    for _ in range(n_init):
        # Initialize centroids randomly
        indices = np.random.choice(features.shape[0], n_clusters, replace=False)
        centroids = features[indices]

        for _ in range(MAX_ITERATIONS):  # Maximum number of iterations
            # Assign points to the nearest centroid
            distances = cdist(features, centroids, 'euclidean')
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(n_clusters)])

            # Check for convergence (if centroids do not change)
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        # Compute inertia (sum of squared distances to the nearest centroid)
        inertia = np.sum((features - centroids[labels]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels
