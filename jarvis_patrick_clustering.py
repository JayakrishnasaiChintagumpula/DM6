"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial.distance import cdist
from matplotlib.backends.backend_pdf import PdfPages

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def calculate_SSE(data, labels):
    total_sse = 0.0
    unique_clusters = np.unique(labels)
    for cluster_id in unique_clusters:
        cluster_points = data[labels == cluster_id]
        if cluster_points.size > 0:
            cluster_center = np.sum(cluster_points, axis=0) / cluster_points.shape[0]
            total_sse += np.sum((cluster_points - cluster_center) ** 2)
    return total_sse

def adjusted_random_index(true_labels, pred_labels):
    def comb(n):
        if n < 2:
            return 0
        return n * (n - 1) / 2

    unique_true, inverse_true = np.unique(true_labels, return_inverse=True)
    unique_pred, inverse_pred = np.unique(pred_labels, return_inverse=True)
    n = len(true_labels)
    contingency_matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=int)

    for i in range(n):
        contingency_matrix[inverse_true[i], inverse_pred[i]] += 1

    sum_total = np.sum(contingency_matrix)
    sum_comb_rows = np.sum([comb(n) for n in np.sum(contingency_matrix, axis=1)])
    sum_comb_cols = np.sum([comb(n) for n in np.sum(contingency_matrix, axis=0)])
    sum_comb_total = np.sum([comb(n) for n in contingency_matrix.flatten()])

    expected_index = sum_comb_rows * sum_comb_cols / comb(sum_total)
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    index = (sum_comb_total - expected_index) / (max_index - expected_index) if max_index != expected_index else 0

    return index

def jarvis_patrick(data, labels, params):
    # Simulated clustering process - this should be replaced with your actual clustering logic
    computed_labels = np.random.randint(0, params['k'], size=len(data))
    cluster_centers = data[np.random.choice(range(len(data)), params['k'], replace=False), :]
    ARI = adjusted_random_index(labels, computed_labels)
    return computed_labels, ARI, cluster_centers

def best_hyperparams(data, labels, k_range, s_min_range, num_trials):
    highest_ARI = -1
    optimal_k = None
    optimal_s_min = None

    for k in k_range:
        for s_min in s_min_range:
            cumulative_ARI = 0
            for _ in range(num_trials):
                params = {'k': k, 's_min': s_min}
                _, ARI_score, _ = jarvis_patrick(data, labels, params)
                cumulative_ARI += ARI_score

            average_ARI = cumulative_ARI / num_trials
            if average_ARI > highest_ARI:
                highest_ARI = average_ARI
                optimal_k = k
                optimal_s_min = s_min

    return optimal_k, optimal_s_min
    
def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """

    computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = 0.

    # A single float
    answers["std_ARIs"] = 0.

    # A single float
    answers["mean_SSEs"] = 0.

    # A single float
    answers["std_SSEs"] = 0.

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
