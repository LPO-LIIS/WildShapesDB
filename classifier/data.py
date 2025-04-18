import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os
import numpy as np
import random

# Normalização e transformações
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
        ),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        normalize,
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)

# Definir o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    """Set seed for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior

class HuggingFaceDataset(torch.utils.data.Dataset):
    """
    Wrapper para adaptar um dataset do Hugging Face ao PyTorch Dataset.
    Aplica transformações nas imagens conforme necessário.
    """

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(
    hf_dataset_name,
    batch_size=32,
    num_workers=12,
    pin_memory=True,
    k_fold=5,
):
    """Cria DataLoaders para validação cruzada com K-Fold."""
    # Carregar dataset do Hugging Face
    dataset = load_dataset(hf_dataset_name)

    # Criar datasets adaptados ao PyTorch
    full_dataset = HuggingFaceDataset(dataset["train"], transform=eval_transform)

    # Se não tiver divisão prévia, faz Stratified K-Fold
    targets = [example["label"] for example in dataset["train"]]

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = eval_transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )

        yield fold, train_loader, val_loader, train_idx, val_idx

def load_best_dataloaders(
    hf_dataset_name,
    batch_size=32,
    num_workers=12,
    pin_memory=True,
    save_dir="dataset_splits"
):
    """Carrega os melhores DataLoaders salvos durante a otimização."""
    
    if not os.path.exists(os.path.join(save_dir, "best_train_indices.pt")):
        raise FileNotFoundError("❌ Não foi encontrado um fold otimizado salvo.")

    # Carregar dataset do Hugging Face
    dataset = load_dataset(hf_dataset_name)

    # Criar datasets adaptados ao PyTorch
    full_dataset = HuggingFaceDataset(dataset["train"], transform=eval_transform)

    # Divisão dos índices para treino/validação/teste
    test_indices = torch.load(os.path.join(save_dir, "test_indices.pt"))

    best_train_indices = torch.load(os.path.join(save_dir, "best_train_indices.pt"), weights_only=False)
    best_val_indices = torch.load(os.path.join(save_dir, "best_val_indices.pt"), weights_only=False)

    print("✅ Carregando o melhor fold encontrado durante otimização...")

    train_dataset = Subset(full_dataset, best_train_indices)
    val_dataset = Subset(full_dataset, best_val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Aplicar transformações específicas
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    # Criar DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader


def split_dataset(hf_dataset_name, test_split=0.1, save_dir="dataset_splits"):
    """
    Divide um dataset do Hugging Face em treino, validação e teste e salva os índices.

    Args:
        hf_dataset_name (str): Nome do dataset no Hugging Face.
        test_split (float): Proporção do conjunto de teste.
        save_dir (str): Diretório onde salvar os índices.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Carregar dataset
    dataset = load_dataset(hf_dataset_name)["train"]
    targets = np.array([example["label"] for example in dataset])

    stratified_split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split, random_state=42
    )

    for train_val_idx, test_idx in stratified_split.split(
        np.zeros(len(targets)), targets
    ):
        train_val_indices = train_val_idx.tolist()
        test_indices = test_idx.tolist()

    # Salvar divisões do dataset
    torch.save(train_val_indices, os.path.join(save_dir, "train_val_indices.pt"))
    torch.save(test_indices, os.path.join(save_dir, "test_indices.pt"))

    print(f"✅ Dataset dividido e salvo em {save_dir}")
