import numpy as np
import umap
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def plot_isolation_forest_clusters(feature_matrix_reduced, anomaly_scores, threshold, plot_name):
    """
    Plots and saves a visualization of Isolation Forest outlier detection.

    Parameters:
    - feature_matrix_reduced (numpy array): Extracted image features after UMAP.
    - anomaly_scores (numpy array): Anomaly scores from Isolation Forest.
    - threshold (float): Threshold for identifying outliers.
    - plot_name (str): Filename for saving the plot.
    """
    plt.figure(figsize=(10, 6))

    # Identifying outliers and regular samples
    outliers = feature_matrix_reduced[anomaly_scores > threshold]
    true_images = feature_matrix_reduced[anomaly_scores <= threshold]

    # Plotting outliers in red
    if outliers.size > 0:
        plt.scatter(
            outliers[:, 0], outliers[:, 1], color="red", alpha=0.8, label="Outliers"
        )

    # Plotting true images in green
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
    plt.title("Isolation Forest Outlier Detection (Outliers in Red)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_name}_isolation_forest.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Isolation Forest plot saved at: {plot_name}_isolation_forest.png")


def auto_tune_isolation_forest_params(feature_matrix):
    """
    Automatically determines the best Isolation Forest parameters based on dataset size.

    Parameters:
    - feature_matrix (numpy array): Extracted image features.

    Returns:
    - n_estimators (int): Number of trees in the Isolation Forest.
    - contamination (float): Estimated proportion of outliers.
    """
    N = feature_matrix.shape[0]  # Number of samples

    # Define n_estimators dynamically (more trees for larger datasets)
    n_estimators = min(200, max(50, N // 10))  # Between 50 and 200 trees

    # Define contamination dynamically (higher for small datasets)
    contamination = min(0.1, max(0.005, 10 / N))  # Between 0.5% and 10%

    return n_estimators, contamination


def detect_outliers_isolation_forest(image_names, feature_matrix, plot_name="plot"):
    """
    Detect outliers using Isolation Forest.

    Parameters:
    - image_names (list): List of image filenames.
    - feature_matrix (numpy array): Extracted image features.

    Returns:
    - List of detected outliers.
    """

    print(f"🔍 Detecting outliers for images using Isolation Forest...")

    # Step 2: Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=10, random_state=42)
    feature_matrix_reduced = umap_reducer.fit_transform(feature_matrix)

    # Step 3: Automatically determine Isolation Forest parameters
    n_estimators, contamination = auto_tune_isolation_forest_params(feature_matrix_reduced)
    print(f"🔧 Auto-selected Isolation Forest parameters: n_estimators={n_estimators}, contamination={contamination:.4f}")

    # Step 4: Apply Isolation Forest for outlier detection
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(feature_matrix_reduced)  # Ajustando o modelo aos dados
    anomaly_scores = -iso_forest.decision_function(feature_matrix_reduced)  # Higher scores = more anomalous

    # Define threshold for outliers
    threshold = np.percentile(anomaly_scores, 90)  # Top 10% most anomalous as outliers

    # Step 5: Identify outliers
    outliers_idx = np.where(anomaly_scores > threshold)[0]

    # Step 6: Plot Isolation Forest clustering
    plot_isolation_forest_clusters(feature_matrix_reduced, anomaly_scores, threshold, plot_name)

    # Return list of "true" outliers
    outliers_list = [image_names[idx] for idx in outliers_idx]
    return outliers_list
