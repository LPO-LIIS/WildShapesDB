import os
import torch
from classifier.training import train_model
from classifier.evaluate import evaluate_model
from classifier.model import Shape2DClassifier
from classifier.data import load_best_dataloaders, set_seed
import torch.optim as optim
import torch._dynamo


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    torch.set_float32_matmul_precision('high')
    
    # Definindo a seed para todos os processos
    set_seed(2108)

    available_gpus = torch.cuda.device_count()
    print(f"Detected {available_gpus} GPUs")

    # Carregar os melhores hiperparâmetros encontrados pela otimização
    best_params_path = os.path.join("optimization_results","best_hyperparameters.pt")
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"❌ Arquivo {best_params_path} não encontrado! Rode a otimização primeiro.")
    
    best_params = torch.load(best_params_path)
    print(f'Melhores parâmetros carregados: {best_params}')

    # Criar os DataLoaders com os melhores parâmetros otimizados
    batch_size = best_params["batch_size"]
    train_loader, val_loader, test_loader = load_best_dataloaders(
        "Horusprg/WildShapes", batch_size=batch_size
    )

    # Determinar o número correto de classes
    num_classes = len(train_loader.dataset.dataset.dataset.features["label"].names)

    print(f"Detected number of classes: {num_classes}")

    # Criar o modelo com o número correto de classes
    model = Shape2DClassifier(num_classes=num_classes)

    # Configurar o otimizador com os melhores hiperparâmetros
    optimizer_name = best_params["optimizer"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]

    optimizer = {
        "Adam": optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        "AdamW": optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    }[optimizer_name]

    # Configurar o scheduler com os melhores hiperparâmetros
    scheduler_name = best_params["scheduler"]
    scheduler = {
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6),
        "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5),
    }[scheduler_name]

    torch.set_float32_matmul_precision('high')


    # Treinar o modelo usando os melhores hiperparâmetros
    accumulation_steps = best_params["accumulation_steps"]
    num_epochs = 150

    # Treinar e avaliar o modelo
    best_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        accumulation_steps=accumulation_steps,
        patience=15
    )

    # Avaliar o modelo e exibir métricas
    metrics = evaluate_model(model, test_loader)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value}")