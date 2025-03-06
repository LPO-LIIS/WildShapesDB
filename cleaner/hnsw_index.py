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
    Detect outliers using FAISS HNSW and Mahalanobis Distance.

    Parameters:
    - index (faiss.IndexHNSWFlat): FAISS index loaded.
    - image_names (list): List of image names.
    - feature_matrix (numpy array): Extracted image features.

    Returns:
    - List of detected outliers.
    """

    print(f"🔍 Detecting outliers for images using FAISS HNSW...")

    # Compute FAISS k-NN search
    k = max(2, int(np.log2(len(image_names))))
    print(f"🔧 Auto-selected HNSW parameters: k={k}")

    # Compute FAISS k-NN search to find nearest neighbors for each feature
    distances, indices = index.search(feature_matrix, k)

    # Compute mean distance to nearest neighbors
    mean_distances = np.mean(distances, axis=1)

    # Compute global mean and covariance matrix of the features
    mean_vector = np.mean(feature_matrix, axis=0)
    covariance_matrix = np.cov(feature_matrix, rowvar=False)

    # Regularizar matriz de covariância para evitar singularidade
    epsilon = 1e-6
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues[eigenvalues < 0] = epsilon  # Evita valores negativos
    covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    inv_covariance = pinv(covariance_matrix)

    # Compute Mahalanobis distances using optimized vectorized operations
    delta = feature_matrix - mean_vector
    mahalanobis_distances = np.sqrt(np.sum(delta @ inv_covariance * delta, axis=1))

    # Calcular o KDE para encontrar o pico da distribuição
    kde = gaussian_kde(mahalanobis_distances)
    x_vals = np.linspace(min(mahalanobis_distances), max(mahalanobis_distances), 1000)
    density = kde(x_vals)
    
    # Encontrar o pico (ponto de maior densidade)
    peak = x_vals[np.argmax(density)]

    # Calcular a média e o desvio padrão após o pico
    distances_after_peak = mahalanobis_distances[mahalanobis_distances > peak]
    if len(distances_after_peak) > 0:
        std_after_peak = np.std(distances_after_peak)
    else:
        std_after_peak = np.std(mahalanobis_distances)  # Fallback

    # Definir threshold como o pico + desvios padrão
    threshold_mahalanobis = peak + std_after_peak

    # Plot distribution before selecting outliers
    plot_mahalanobis_distribution(mahalanobis_distances, threshold_mahalanobis, plot_name)

    # Define outliers using Mahalanobis threshold
    outliers_idx = np.where(mahalanobis_distances > threshold_mahalanobis)[0]

    # Move outliers to separate directory
    outliers_list = []
    for idx in tqdm(outliers_idx, desc="📦 Getting outliers"):
        outliers_list.append(image_names[idx])
    return outliers_list