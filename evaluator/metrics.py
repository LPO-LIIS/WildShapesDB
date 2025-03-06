import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
import itertools
import random

def pca_variance(features, n_components=10):
    """Compute PCA and return variance explained ratio."""

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)

    # Retornar a matriz reduzida e a variância explicada corretamente
    return reduced, np.sum(pca.explained_variance_ratio_)

def centroid_distance(features):
    """Computes the Euclidean distance of each image feature vector to the dataset centroid."""
    
    # Calcular centroide sem normalização
    centroid = np.mean(features, axis=0)
    
    # Calcular distâncias Euclidianas
    distances = cdist(features, [centroid], metric="euclidean").flatten()

    return distances


def compute_ssim(image_paths, num_samples=20, num_workers=4):
    """Calcula a similaridade média entre pares de imagens usando SSIM.
    
    Parâmetros:
    - image_paths (list): Lista de caminhos das imagens.
    - num_samples (int): Número de imagens amostradas aleatoriamente.

    Retorna:
    - mean_ssim (float): Média das similaridades calculadas.
    - std_ssim (float): Desvio padrão das similaridades.
    """
    if len(image_paths) == 0:  # Correct way to check if the list is empty
        return 0, 0  

    # Seleciona um subconjunto aleatório de imagens (até `num_samples`)
    sample_size = min(len(image_paths), num_samples)
    selected_images = random.sample(image_paths, sample_size)

    # Generate all possible pairs within the subset
    pairs = list(itertools.combinations(selected_images, 2))

    def process_pair(pair):
        # Read images
        p1, p2 = pair
        img1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)

        # Ensure images are loaded properly
        if img1 is None or img2 is None:
            return None

        # Resize to the same dimensions if necessary
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Normalize images (convert to float and scale between 0 and 1)
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # Compute SSIM
        similarity = ssim(img1, img2, data_range=1.0)  # Explicitly set data range

        return similarity

    # Executa processamento paralelo das comparações
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        similarities = list(filter(None, executor.map(process_pair, pairs)))

    if similarities:
        return np.mean(similarities), np.std(similarities)
    return 0, 0  # Handle edge case where no valid images were compared
