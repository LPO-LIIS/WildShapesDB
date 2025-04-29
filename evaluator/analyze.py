from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from cleaner.feature_extractor import extract_features_from_images
from evaluator.metrics import pca_variance, centroid_distance, compute_ssim


def analyze_class(class_path):
    """Analyzes a single class and returns computed metrics."""
    image_paths = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))

    if len(image_paths) < 10:
        print(f"⚠️ Class {class_path.name} has too few images for meaningful analysis.")
        return None

    print(f"🔍 Analyzing class: {class_path.name}")
    
    feature_matrix, features_dict = extract_features_from_images(str(class_path))

    # Compute all metrics
    pca_reduced, pca_var = pca_variance(feature_matrix)
    distances = centroid_distance(feature_matrix)
    mean_ssim, std_ssim = compute_ssim(image_paths, 100)

    return {
        "class_name": class_path.name,
        "pca_variance": pca_var,
        "centroid_distances": distances,
        "mean_centroid_distance": np.mean(distances),
        "std_centroid_distance": np.std(distances),
        "mean_ssim": mean_ssim,
        "std_ssim": std_ssim,
        "pca_reduced": pca_reduced,
    }


def analyze_dataset(dataset_path, output_path, num_workers=4):
    """
    Analyzes all classes within a dataset and generates global statistical summaries.

    Parameters:
        dataset_path (str or Path): Path to the dataset directory containing class subfolders.
        output_path (str or Path): Path where the generated plots will be saved.
        num_workers (int): Number of threads to use for parallel processing.

    Output:
        - Saves plots in the specified directory.
        - Prints global dataset statistics.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    class_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

    class_results = []
    for class_folder in class_folders:
        class_results.append(analyze_class(class_folder))

    # Filter out None results (classes with too few images)
    class_results = [res for res in class_results if res is not None]

    if not class_results:
        print("❌ No valid data found for analysis.")
        return

    # Extract and structure metrics
    metrics = {
        "class_names": [res["class_name"] for res in class_results],
        "pca_variances": [res["pca_variance"] for res in class_results],
        "centroid_distances": [res["centroid_distances"] for res in class_results],
        "mean_ssim_values": [res["mean_ssim"] for res in class_results],
        "std_ssim_values": [res["std_ssim"] for res in class_results],
    }

    # Generate plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    # PCA Variance per Class
    axes[0].bar(metrics["class_names"], metrics["pca_variances"], color="royalblue", alpha=0.8, edgecolor="black")
    axes[0].set_title("PCA Variance per Class", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Class", fontsize=14)
    axes[0].set_ylabel("PCA Variance", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Centroid Distance Distribution (Boxplot)
    box = axes[1].boxplot(metrics["centroid_distances"], patch_artist=True, labels=metrics["class_names"],
                        boxprops=dict(facecolor="lightblue", color="black"),
                        capprops=dict(color="black"),
                        whiskerprops=dict(color="black"),
                        flierprops=dict(markeredgecolor="red", alpha=0.6),
                        medianprops=dict(color="black"))
    axes[1].set_title("Centroid Distance Distribution", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Class", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("Distance to Centroid", fontsize=14)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # SSIM per Class
    axes[2].bar(
        metrics["class_names"],
        metrics["mean_ssim_values"],
        yerr=metrics["std_ssim_values"],
        capsize=5,
        color="mediumseagreen",
        alpha=0.8,
        edgecolor="black")
    axes[2].set_title("Structural Similarity Index (SSIM) per Class", fontsize=16, fontweight='bold')
    axes[2].set_xlabel("Class", fontsize=14)
    axes[2].set_ylabel("Mean SSIM", fontsize=14)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the plot to file
    plot_path = os.path.join(output_path, "dataset_analysis.png")
    plt.savefig(plot_path, dpi=300)
    print(f"📊 Analysis plots saved to: {plot_path}")

    plt.close()


    # 📊 Compute global dataset statistics
    global_stats = {
        "mean_pca_variance": np.mean(metrics["pca_variances"]),
        "mean_centroid_distance": np.mean([np.mean(res["centroid_distances"]) for res in class_results]),
        "std_centroid_distance": np.mean([np.std(res["centroid_distances"]) for res in class_results]),
        "mean_ssim_global": np.mean(metrics["mean_ssim_values"]),
        "std_ssim_global": np.mean(metrics["std_ssim_values"]),
    }

    # Print dataset summary
    print("\n📊 Global Dataset Statistics:")
    print(f"   Mean PCA Variance: {global_stats['mean_pca_variance']:.4f}")
    print(f"   Mean Centroid Distance: {global_stats['mean_centroid_distance']:.4f} (± {global_stats['std_centroid_distance']:.4f})")
    print(f"   Mean SSIM: {global_stats['mean_ssim_global']:.4f} (± {global_stats['std_ssim_global']:.4f})")