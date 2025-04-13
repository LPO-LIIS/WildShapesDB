import faiss
import numpy as np
import os
import json
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde


def create_faiss_hnsw_index(features_dict, output_path="output/", hnsw_layers=32):
    """
    Cria e salva um índice FAISS HNSW para busca eficiente de imagens similares.

    Parâmetros:
    - features_dict (dict): Dicionário {nome_da_imagem: feature}.
    - output_path (str): Caminho do arquivo onde o índice FAISS será salvo.
    - hnsw_layers (int): Número de camadas no grafo HNSW (padrão: 32).

    Retorno:
    - index (faiss.IndexHNSWFlat): Índice FAISS criado.
    """

    # Criar a pasta de saída se não existir
    os.makedirs(output_path, exist_ok=True)

    # Converter features_dict para matriz NumPy
    image_names = list(features_dict.keys())  # Lista com os nomes das imagens
    feature_matrix = np.array(list(features_dict.values()), dtype="float32")

    # Normalizar features antes de indexação
    scaler = StandardScaler()
    feature_matrix_normalized = scaler.fit_transform(feature_matrix)

    # Normalizar vetores para FAISS (Inner Product Search)
    faiss.normalize_L2(feature_matrix_normalized)

    # Definir a dimensão das features (exemplo: 2048 para ResNet50)
    dimension = feature_matrix.shape[1]

    # Criar um índice HNSW para busca eficiente
    index = faiss.IndexHNSWFlat(dimension, hnsw_layers)

    # Adicionar todas as features ao índice
    index.add(feature_matrix_normalized)

    # Criar diretório de saída caso não exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salvar o índice FAISS para uso futuro
    faiss.write_index(index, f"{output_path}/features_hnsw.faiss")

    # Salvar o mapeamento nome_da_imagem → índice no FAISS
    json_output_path = f"{output_path}/features_metadata.json"
    with open(json_output_path, "w") as f:
        json.dump(image_names, f)

    print(f"✅ Índice FAISS criado e salvo em {output_path}")
    print(f"✅ Mapeamento de imagens salvo em {json_output_path}")

    return index, image_names  # Retorna o índice FAISS e a lista de imagens


def plot_mahalanobis_distribution(mahalanobis_distances, threshold_mahalanobis, name="plot"):
    """
    Plots and saves a distribution curve of Mahalanobis distances to visualize outliers.

    Parameters:
    - mahalanobis_distances (numpy array): Mahalanobis distances calculated for each image.
    - threshold_mahalanobis (float): Threshold for defining outliers.
    - name (str): Name of the plot file.
    """

    # Fit a chi-square distribution for visualizationw
    df = min(len(mahalanobis_distances) - 1, 100)  # Evita df muito alto
    x = np.linspace(0, np.max(mahalanobis_distances), 1000)
    y = chi2.pdf(x, df)

    # Create histogram of Mahalanobis distances
    plt.figure(figsize=(10, 5))
    sns.histplot(mahalanobis_distances, kde=True, bins=30, stat="density", alpha=0.6, label="Mahalanobis Distance Distribution")

    # Mark threshold for outliers
    plt.axvline(threshold_mahalanobis, color="red", linestyle="dashed", label=f"Outlier Threshold ({threshold_mahalanobis:.2f})")

    # Add text annotations for threshold
    plt.text(threshold_mahalanobis + 0.2, max(y) * 0.5, f"Threshold Distance:\n{threshold_mahalanobis:.3f}", 
             color="red", fontsize=12, fontweight="bold", bbox=dict(facecolor='white', alpha=0.6))

    # Labels and title
    plt.xlabel("Mahalanobis Distance", fontsize=12, fontweight="bold")
    plt.ylabel("Density", fontsize=12, fontweight="bold")
    plt.title("Mahalanobis Distance Distribution for Outlier Detection", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{name}_mahalanobis_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"📊 Mahalanobis distribution plot saved at: {name}_mahalanobis_distribution.png")

def detect_outliers_hnsw(
    index,
    image_names,
    feature_matrix,
    plot_name="plot",
):
    """
    Detect outliers using FAISS HNSW and **Local Mahalanobis Distance** (based on nearest neighbors).

    Parameters:
    - index (faiss.IndexHNSWFlat): FAISS index loaded.
    - image_names (list): List of image names.
    - feature_matrix (numpy array): Extracted image features.

    Returns:
    - List of detected outliers.
    """

    print(f"🔍 Detecting outliers for images using FAISS HNSW + Local Mahalanobis...")

    # Step 1: Compute FAISS k-NN search (to find neighbors for each point)
    k = max(2, int(np.log2(len(image_names))))
    print(f"🔧 Auto-selected HNSW parameters: k={k}")

    distances, indices = index.search(feature_matrix, k)

    # Step 2: Compute Local Mahalanobis Distance for each point
    mahalanobis_distances = np.zeros(len(feature_matrix))

    for i in tqdm(range(len(feature_matrix)), desc="📊 Computing Local Mahalanobis"):
        # Get the k-nearest neighbors (including the point itself)
        neighbors = feature_matrix[indices[i]]
        
        # Compute local mean and covariance
        local_mean = np.mean(neighbors, axis=0)
        local_cov = np.cov(neighbors, rowvar=False)
        
        # Regularize covariance matrix (avoid singularity)
        epsilon = 1e-6
        eigenvalues, eigenvectors = np.linalg.eigh(local_cov)
        eigenvalues[eigenvalues < 0] = epsilon
        local_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        inv_local_cov = pinv(local_cov)
        
        # Compute Mahalanobis distance for this point
        delta = feature_matrix[i] - local_mean
        mahalanobis_distances[i] = np.sqrt(delta.T @ inv_local_cov @ delta)

    # Step 3: Find outliers using KDE peak detection
    kde = gaussian_kde(mahalanobis_distances)
    x_vals = np.linspace(min(mahalanobis_distances), max(mahalanobis_distances), 1000)
    density = kde(x_vals)
    peak = x_vals[np.argmax(density)]

    # Threshold = peak + std of distances after the peak
    distances_after_peak = mahalanobis_distances[mahalanobis_distances > peak]
    std_after_peak = np.std(distances_after_peak) if len(distances_after_peak) > 0 else np.std(mahalanobis_distances)
    threshold = peak + std_after_peak

    # Step 4: Identify outliers
    outliers_idx = np.where(mahalanobis_distances > threshold)[0]
    outliers_list = [image_names[idx] for idx in outliers_idx]

    # (Optional) Plot distribution
    plot_mahalanobis_distribution(mahalanobis_distances, threshold, plot_name)

    return outliers_list