import torch
from classifier.training import train_model
from classifier.evaluate import evaluate_model
from classifier.model import Shape2DClassifier
from classifier.data import create_dataloaders, set_seed

if __name__ == "__main__":
    # Definindo a seed para todos os processos
    set_seed(2108)

    available_gpus = torch.cuda.device_count()
    print(f"Detected {available_gpus} GPUs")
    train_loader, val_loader, test_loader = create_dataloaders("WildShapesDataset/images", batch_size=128, augment=True)

    # Determinar o número correto de classes
    num_classes = len(train_loader.dataset.classes)

    print(f"Detected number of classes: {num_classes}")

    # Criar o modelo com o número correto de classes
    model = Shape2DClassifier(num_classes=num_classes)

    # Treinar e avaliar o modelo
    train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3)

    # Avaliar o modelo e exibir métricas
    metrics = evaluate_model(model, test_loader)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")