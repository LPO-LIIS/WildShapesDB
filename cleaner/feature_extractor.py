import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """Custom dataset to load and process images"""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = self.transform(img)
        return img, os.path.basename(image_path)


def extract_features_from_images(image_dir, batch_size=128):
    """
    Extrai features de imagens em um diretório usando ResNet50 e salva em um arquivo NumPy.

    Parâmetros:
    - image_dir (str): Caminho do diretório contendo as imagens.
    - batch_size (int): Número de imagens processadas simultaneamente.

    Retorno:
    - features_dict (dict): Dicionário {nome_da_imagem: feature}
    """

    # Configurar o dispositivo (GPU se disponível)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar o modelo ResNet50 pré-treinado e remover a última camada (FC layer)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove a última camada
    model = model.to(device)
    model.eval()  # Modo de inferência

    # Transformações para pré-processamento das imagens
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Redimensiona a imagem
            transforms.ToTensor(),  # Converte para tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalização padrão
        ]
    )

    # Lista de imagens no diretório (suporta JPG e PNG)
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_paths:
        print("Nenhuma imagem encontrada no diretório especificado.")
        return {}

    # Criar Dataset e DataLoader para processar em batch
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Dicionário para armazenar as features associadas aos nomes das imagens
    features_dict = {}

    # Processamento em batch com tqdm para barra de progresso
    all_features = []
    image_names = []

    with torch.no_grad():  # Desliga gradientes para economizar memória
        for images, names in tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Extracting Features from {image_dir}",
        ):
            images = images.to(device)  # Enviar batch para GPU

            # Extração das features
            features = model(images)  # Saída shape: (batch_size, 2048, 1, 1)
            features = features.view(
                features.size(0), -1
            )  # Flatten para (batch_size, 2048)

            # Converter para NumPy e armazenar
            all_features.append(features.cpu().numpy())
            image_names.extend(names)

    # Concatenar todas as features extraídas
    feature_matrix = np.vstack(all_features)

    # Criar dicionário com nomes das imagens e suas respectivas features
    features_dict = {
        name: feature for name, feature in zip(image_names, feature_matrix)
    }

    return feature_matrix, features_dict
