import numpy as np
import hdbscan
import umap
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_hdbscan_clusters(feature_matrix_reduced, clusters, plot_name):
    """
    Plots and saves a visualization of HDBSCAN clustering results, highlighting outliers.

    Parameters:
    - feature_matrix_reduced (numpy array): Extracted image features.
    - clusters (numpy array): Cluster assignments from HDBSCAN.
    - plot_name (str): Filename for saving the plot.
    """
    plt.figure(figsize=(10, 6))

    # Identifying outliers and true images
    outliers = feature_matrix_reduced[clusters == -1]
    true_images = feature_matrix_reduced[clusters != -1]

    # Plotting outliers in red
    if outliers.size > 0:
        plt.scatter(
            outliers[:, 0], outliers[:, 1], color="red", alpha=0.8, label="Outliers"
        )

    # Plotting true images in blue
    if true_images.size > 0:
        plt.scatter(
            true_images[:, 0],
            true_images[:, 1],
            color="green",
            alpha=0.6,
            label="Regular Sample",
        )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("HDBSCAN Cluster Visualization (Outliers in Red)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_name}_hdbscan_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 HDBSCAN cluster plot saved at: {plot_name}_hdbscan_clusters.png")


def auto_tune_hdbscan_params(feature_matrix):
    """
    Automatically determines the best HDBSCAN parameters based on dataset size.

    Parameters:
    - feature_matrix (numpy array): Extracted image features.

    Returns:
    - min_cluster_size (int)
    - min_samples (int)
    """
    N = feature_matrix.shape[0]  # Number of samples

    # Dynamically adjust min_cluster_size and min_samples
    min_cluster_size = max(2, N // 50)  # At least 2, max 2% of dataset
    min_samples = max(1, int(np.sqrt(N)))  # Square root of dataset size

    return min_cluster_size, min_samples


def detect_outliers_hdbscan(
    image_names, feature_matrix, plot_name="plot"
):
    """
    Detect outliers using HDBSCAN clustering algorithm.

    Parameters:
    - image_names (list): List of image filenames.
    - feature_matrix (numpy array): Extracted image features.
    - output_outliers (str): Directory where outlier images will be moved.

    Returns:
    - List of detected outliers.
    """

    print(f"🔍 Detecting outliers for images using HDBSCAN...")

    # Step 2: Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=10, random_state=42, metric="cosine")
    feature_matrix_reduced = umap_reducer.fit_transform(feature_matrix)

    # Step 3: Automatically determine HDBSCAN parameters
    min_cluster_size, min_samples = auto_tune_hdbscan_params(feature_matrix_reduced)
    print(f"🔧 Auto-selected HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # Step 4: Apply HDBSCAN for clustering
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    clusters = hdb.fit_predict(feature_matrix_reduced)

    # Step 5: Identify outliers (-1 cluster)
    outliers_idx = np.where(clusters == -1)[0]

    # Step 6: Plot HDBSCAN clustering
    plot_hdbscan_clusters(feature_matrix_reduced, clusters, plot_name)

    # Return list of "true" outliers
    outliers_list = [image_names[idx] for idx in outliers_idx]
    return outliers_list
