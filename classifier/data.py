import torch
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed: int = 42):
    """Set seed for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensure deterministic behavior

class CachedDataset(Dataset):
    def __init__(self, dataset, transform=None, cache_dir="cache", use_disk_cache=False):
        """
        Args:
            dataset (Dataset): O dataset original (por exemplo, Subset).
            transform (callable): Transformações a serem aplicadas às imagens.
            cache_dir (str): Diretório para armazenar o cache em disco.
            use_disk_cache (bool): Se True, usa cache em disco; caso contrário, usa cache em memória.
        """
        self.dataset = dataset
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_disk_cache = use_disk_cache
        self.cache = {}  # Cache em memória

        if use_disk_cache:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Verifica se o item já está em cache
        if idx in self.cache:
            return self.cache[idx]

        # Carrega o item do dataset original
        image, label = self.dataset[idx]

        # Aplica as transformações, se houver
        if self.transform:
            image = self.transform(image)

        # Armazena no cache
        if self.use_disk_cache:
            cache_path = os.path.join(self.cache_dir, f"{idx}.pt")
            torch.save((image, label), cache_path)
        else:
            self.cache[idx] = (image, label)

        return image, label

def create_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    k_fold=5,
    load_best_fold=False,
):
    """
    Creates DataLoaders for training, validation, and testing.

    This function loads dataset indices from precomputed splits and applies transformations.
    If `load_best_fold=True`, it loads the best training-validation split found during hyperparameter optimization.
    Otherwise, it performs k-fold cross-validation on the training-validation set.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Number of images per batch (default: 32).
        num_workers (int): Number of subprocesses for data loading (default: 4).
        pin_memory (bool): If True, enables pinned memory for faster GPU training (default: True).
        k_fold (int): Number of folds for stratified k-fold cross-validation (default: 5).
        load_best_fold (bool): Whether to load the best training-validation split from saved indices (default: False).

    Returns:
        - If `load_best_fold=True`:
            tuple: (train_loader, val_loader, test_loader)
        - If `load_best_fold=False`:
            generator yielding (fold, train_loader, val_loader, train_indices, val_indices, test_loader)
    """
    save_dir = "dataset_splits"

    # Load precomputed dataset indices
    train_val_indices = torch.load(os.path.join(save_dir, "train_val_indices.pt"), weights_only=True)
    test_indices = torch.load(os.path.join(save_dir, "test_indices.pt"), weights_only=True)

    # Define data transformations
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
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.1), ratio=(0.5, 2.0), value=0
            ),
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

    dataset = datasets.ImageFolder(root=data_dir)

    # Create test dataset using precomputed indices
    test_dataset = Subset(dataset, test_indices)
    test_dataset.dataset.transform = eval_transform  # Apply evaluation transformations
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if load_best_fold:
        # Load the best fold indices found during optimization
        best_train_indices = torch.load(os.path.join(save_dir, "best_train_indices.pt"))
        best_val_indices = torch.load(os.path.join(save_dir, "best_val_indices.pt"))

        print("✅ Loading the best fold found during optimization...")

        train_dataset = Subset(dataset, best_train_indices)
        val_dataset = Subset(dataset, best_val_indices)
        train_dataset = CachedDataset(Subset(dataset, best_train_indices), transform=train_transform)
        val_dataset = CachedDataset(Subset(dataset, best_val_indices), transform=eval_transform)

        # Apply transformations
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = eval_transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    # Perform stratified k-fold cross-validation on the training-validation set
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    targets = [dataset.targets[i] for i in train_val_indices]

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, targets)):
        # Cria os datasets de treino e validação com cache
        train_dataset = CachedDataset(
            Subset(dataset, [train_val_indices[i] for i in train_idx]),
            transform=train_transform,
            use_disk_cache=False,
        )
        val_dataset = CachedDataset(
            Subset(dataset, [train_val_indices[i] for i in val_idx]),
            transform=eval_transform,
            use_disk_cache=False,
        )

        # Apply transformations
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = eval_transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        yield fold, train_loader, val_loader, [
            train_val_indices[i] for i in train_idx
        ], [train_val_indices[i] for i in val_idx], test_loader


def split_dataset(data_dir, test_split=0.1, save_dir="dataset_splits"):
    """
    Splits the dataset into Training+Validation (90%) and Testing (10%) using stratified sampling and saves the indices.

    This function ensures that the class distribution remains balanced across training/validation and testing sets.

    Args:
        data_dir (str): Path to the dataset directory.
        test_split (float): Proportion of the dataset to be used for testing (default: 0.1).
        save_dir (str): Directory where the dataset splits will be saved (default: "dataset_splits").

    Saves:
        - `train_val_indices.pt`: Indices for the Training+Validation set (90% of the dataset).
        - `test_indices.pt`: Indices for the Testing set (10% of the dataset).
    """
    os.makedirs(save_dir, exist_ok=True)

    dataset = ImageFolder(root=data_dir)
    targets = np.array(dataset.targets)  # Extract class labels from dataset

    stratified_split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_split, random_state=42
    )

    for train_val_idx, test_idx in stratified_split.split(
        np.zeros(len(targets)), targets
    ):
        train_val_indices = train_val_idx.tolist()
        test_indices = test_idx.tolist()

    # Save the dataset splits
    torch.save(train_val_indices, os.path.join(save_dir, "train_val_indices.pt"))
    torch.save(test_indices, os.path.join(save_dir, "test_indices.pt"))

    print(f"✅ Dataset successfully split and saved in {save_dir}")
