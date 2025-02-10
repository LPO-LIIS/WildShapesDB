import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    """ Dataset personalizado para carregar e processar imagens. """
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        return img, os.path.basename(image_path)  # Retorna imagem transformada e nome

def extract_features_from_images(image_dir, output_path="features/", batch_size=128):
    """
    Extrai features de imagens em um diretório usando ResNet50 e salva em um arquivo NumPy.

    Parâmetros:
    - image_dir (str): Caminho do diretório contendo as imagens.
    - output_path (str): Caminho do arquivo de saída para salvar as features.
    - batch_size (int): Número de imagens processadas simultaneamente.

    Retorno:
    - features_dict (dict): Dicionário {nome_da_imagem: feature}
    """

    # Criar a pasta de saída se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Configurar o dispositivo (GPU se disponível)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar o modelo ResNet50 pré-treinado e remover a última camada (FC layer)
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove a última camada
    model = model.to(device)
    model.eval()  # Modo de inferência

    # Transformações para pré-processamento das imagens
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona a imagem
        transforms.ToTensor(),  # Converte para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização padrão
    ])

    # Lista de imagens no diretório (suporta JPG e PNG)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_paths:
        print("Nenhuma imagem encontrada no diretório especificado.")
        return {}

    # Criar Dataset e DataLoader para processar em batch
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Dicionário para armazenar as features associadas aos nomes das imagens
    features_dict = {}

    # Processamento em batch com tqdm para barra de progresso
    all_features = []
    image_names = []
    
    with torch.no_grad():  # Desliga gradientes para economizar memória
        for images, names in tqdm(dataloader, total=len(dataloader), desc="Extracting Features"):
            images = images.to(device)  # Enviar batch para GPU

            # Extração das features
            features = model(images)  # Saída shape: (batch_size, 2048, 1, 1)
            features = features.view(features.size(0), -1)  # Flatten para (batch_size, 2048)

            # Converter para NumPy e armazenar
            all_features.append(features.cpu().numpy())
            image_names.extend(names)

    # Concatenar todas as features extraídas
    feature_matrix = np.vstack(all_features)

    # Criar dicionário com nomes das imagens e suas respectivas features
    features_dict = {name: feature for name, feature in zip(image_names, feature_matrix)}

    # Salvar as features como arquivo NumPy
    np.save(f"{output_path}/features.npy", feature_matrix)

    # Salvar o mapeamento de nomes das imagens para um arquivo JSON
    json_output_path = f"{output_path}/features_metadata.json"
    with open(json_output_path, "w") as f:
        json.dump(image_names, f)

    print(f"✅ Features extraídas e salvas em {output_path}")
    print(f"✅ Metadados (nomes das imagens) salvos em {json_output_path}")

    return features_dict