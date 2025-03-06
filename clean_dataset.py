import os
import json
import matplotlib
import numpy as np
import shutil
from tqdm import tqdm
from cleaner.feature_extractor import extract_features_from_images
from cleaner.hnsw_index import create_faiss_hnsw_index, detect_outliers_hnsw
from cleaner.hdbscan import detect_outliers_hdbscan
from cleaner.isolation_forest import detect_outliers_isolation_forest
from evaluator.healthcheck import check_images
from evaluator.analyze import analyze_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")


if __name__ == "__main__":
    image_path = "WildShapesDataset/images/"
    feature_path = "WildShapesDataset/features/"
    indexes_path = "WildShapesDataset/indexes/"
    outliers_path = "WildShapesDataset/outliers/"
    plot_path = "plots/"
    os.makedirs(plot_path, exist_ok=True)

    # Verificar a integridade do dataset
    #check_images(image_path)

    # Avaliar a qualidade do dataset inicial
    print("\n\n📊 Starting dataset evaluation...")
    analyze_dataset(image_path, os.path.join(plot_path, "initial_dataset/"))

    print("\n\n🚀 Starting outlier removal...")
    for folder in os.listdir(image_path):
        print(f"📂 Processing folder: {folder}")

        # Extrair features
        feature_matrix, features_dict = extract_features_from_images(
            os.path.join(image_path, folder)
        )
        
        # Criar a pasta de saída se não existir
        os.makedirs(os.path.join(feature_path, folder), exist_ok=True)

        # Salvar as features como arquivo NumPy
        np.save(f"{os.path.join(feature_path, folder)}/features.npy", feature_matrix)

        print(f"✅ Extracted features and saved to {os.path.join(feature_path, folder)}")

        # Criar índice FAISS
        index, image_names = create_faiss_hnsw_index(
            features_dict, os.path.join(indexes_path, folder)
        )

        # Salvar o mapeamento de nomes das imagens para um arquivo JSON
        json_output_path = (
            f"{os.path.join(feature_path, folder)}/faiss_metadata.json"
        )
        with open(json_output_path, "w") as f:
            json.dump(image_names, f)

        print(f"✅ Metadata (image names) from FAISS saved to {json_output_path}")

        outliers_hnsw_list = detect_outliers_hnsw(
            index,
            image_names,
            feature_matrix,
            os.path.join(plot_path, folder)
        )
        print(
            f"🚨 Outliers detected for {folder} with HNSW+Mahalanobis: {len(outliers_hnsw_list)}"
        )

        outliers_hdbscan_list = detect_outliers_hdbscan(
            image_names,
            feature_matrix,
            os.path.join(plot_path, folder)
        )
        print(
            f"🚨 Outliers detected for {folder} with HDBSCAN: {len(outliers_hdbscan_list)}"
        )

        outliers_isolation_list = detect_outliers_isolation_forest(
            image_names,
            feature_matrix,
            os.path.join(plot_path, folder)
        )
        print(
            f"🚨 Outliers detected for {folder} with Isolation Forest: {len(outliers_isolation_list)}"
        )

        # Converter listas para conjuntos
        outliers_hnsw_set = set(outliers_hnsw_list)
        outliers_hdbscan_set = set(outliers_hdbscan_list)
        outliers_isolation_set = set(outliers_isolation_list)

        # Encontrar interseção (elementos em comum)
        common_outliers = outliers_hnsw_set.union(outliers_hdbscan_set)
        common_outliers = common_outliers.union(outliers_isolation_set)

        # Exibir os resultados
        print(f"🔍 Total outliers in common: {len(common_outliers)}")

        for common_outlier in tqdm(
            common_outliers, desc=f"📦 Moving outliers to folder: {folder}"
        ):
            os.makedirs(os.path.join(outliers_path, folder), exist_ok=True)
            shutil.move(
                os.path.join(image_path, folder, common_outlier),
                os.path.join(outliers_path, folder),
            )
        
    # Avaliar a qualidade do dataset após a remoção de outliers
    print("\n\n📊 Starting dataset evaluation after outlier removal...")
    analyze_dataset(image_path, os.path.join(plot_path, "cleaned_dataset/"))
