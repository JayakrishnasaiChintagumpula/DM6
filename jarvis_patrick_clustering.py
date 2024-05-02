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
    k = params_dict['k']  
    s_min = params_dict['s_min']
    num_samples = len(data)
    computed_labels = np.zeros(num_samples, dtype=np.int32)
    
    # Normalizing data to have mean 0 and standard deviation 1 for each feature
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Calculate the distance matrix for the normalized data
    distance_matrix = cdist(normalized_data, normalized_data, 'euclidean')

    for i in range(num_samples):
        # Get distances to all other points and sort them to find the nearest neighbors
        distances = distance_matrix[i]
        nearest_neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self by starting from index 1

        # Count how many times each label appears among the k-nearest neighbors
        neighbor_label_counts = np.bincount(labels[nearest_neighbor_indices], minlength=np.max(labels) + 1)

        # Find the label with the maximum count (the dominant label)
        dominant_label = np.argmax(neighbor_label_counts)
        dominant_proportion = neighbor_label_counts[dominant_label] / k

        # Assign the dominant label if it meets the similarity threshold
        if dominant_proportion >= s_min:
            computed_labels[i] = dominant_label

    # Calculate the Adjusted Rand Index and SSE for the assigned labels
    ARI = adjusted_random_index(labels, computed_labels)
    SSE = calculate_SSE(data, computed_labels)
    
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
    clust_data = np.load("question1_cluster_data.npy")
    clust_labels = np.load("question1_cluster_labels.npy")

    # Select a subset of data for hyperparameter tuning
    random_indices = np.random.choice(len(clust_data), size=5000, replace=False)
    data_subset = clust_data[random_indices][:1000]
    labels_subset = clust_labels[random_indices][:1000]

    # Define hyperparameter ranges
    k_range = [3, 4, 5, 6, 7, 8]
    s_min_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Determine the best hyperparameters
    num_trials = 10
    best_k, best_s_min = best_hyperparams(data_subset, labels_subset, k_range, s_min_range, num_trials)

    # Set parameters
    params_dict = {'k': best_k, 's_min': best_s_min}
    groups = {}
    plots_values = {}

    # Perform clustering on different data segments
    for i in range(5):
        data_slice = clust_data[i * 1000: (i + 1) * 1000]
        labels_slice = clust_labels[i * 1000: (i + 1) * 1000]
        
        computed_labels, sse, ari = jarvis_patrick(data_slice, labels_slice, params_dict)
        groups[i] = {"smin": best_s_min, "k": best_k, "ARI": ari, "SSE": sse}
        plots_values[i] = {"computed_labels": computed_labels, "ARI": ari, "SSE": sse}

    # Determine the segment with the highest ARI and the lowest SSE
    highest_ari_index = max(plots_values, key=lambda x: plots_values[x]['ARI'])
    lowest_sse_index = min(plots_values, key=lambda x: plots_values[x]['SSE'])

    # Create PDF for plots
    pdf_pages = PdfPages("plots_for_jarvis_patrick_clustering.pdf")

    # Plotting the results for the highest ARI
    plt.figure(figsize=(8, 6))
    plot_ARI = plt.scatter(clust_data[highest_ari_index * 1000: (highest_ari_index + 1) * 1000, 0],
                        clust_data[highest_ari_index * 1000: (highest_ari_index + 1) * 1000, 1],
                        c=plots_values[highest_ari_index]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for Dataset {highest_ari_index} (Highest ARI) with k={best_k}, s_min={best_s_min}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='ID')
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    # Plotting the results for the lowest SSE
    plt.figure(figsize=(8, 6))
    plot_SSE = plt.scatter(clust_data[lowest_sse_index * 1000: (lowest_sse_index + 1) * 1000, 0],
                        clust_data[lowest_sse_index * 1000: (lowest_sse_index + 1) * 1000, 1],
                        c=plots_values[lowest_sse_index]["computed_labels"], cmap='viridis')
    plt.title(f'Clustering for Dataset {lowest_sse_index} (Lowest SSE) with k={best_k}, s_min={best_s_min}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster ID')
    plt.grid(True)
    pdf_pages.savefig()
    plt.close()

    pdf_pages.close()


    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = groups[0]["SSE"]

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    #plot_ARI = plt.scatter([1,2,3], [4,5,6])
    #plot_SSE = plt.scatter([1,2,3], [4,5,6])
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
