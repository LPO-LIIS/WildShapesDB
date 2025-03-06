import torch
import torch.optim as optim
from classifier.model import Shape2DClassifier
from classifier.training import train_model
from classifier.data import create_dataloaders
from classifier.evaluate import evaluate_model
import os


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization using Stratified K-Fold cross-validation.
    Evaluates models using weighted F1 Score.

    Returns:
        float: The best F1 Score obtained during validation.
    """
    data_dir = "WildShapesDataset/images"
    save_dir = "dataset_splits"
    os.makedirs(save_dir, exist_ok=True)

    # Define the hyperparameter search space
    batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512]
    )  # Added 16 and 256 for more variability
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-6, 1e-1, log=True
    )  # Wider range for learning rate tuning
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"]
    )  # Added RMSprop and Adagrad
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["CosineAnnealingLR", "StepLR"]
    )  # Added ReduceLROnPlateau
    accumulation_steps = trial.suggest_categorical(
        "accumulation_steps", [1, 2, 4, 8]
    )  # Added 8 for more gradient accumulation

    # K-Fold Validation Setup
    kfold = 5  # 5-Fold Stratified
    best_f1 = 0.0
    best_fold = None
    best_train_indices = None
    best_val_indices = None

    for (
        fold,
        train_loader,
        val_loader,
        train_indices,
        val_indices,
        _,
    ) in create_dataloaders(data_dir, batch_size=batch_size, k_fold=kfold):
        print(f"🔄 Training Fold {fold+1}/{kfold}")

        # Initialize the model
        model = Shape2DClassifier(num_classes=9).cuda()

        # Select optimizer
        momentum = (
            trial.suggest_float("momentum", 0.7, 0.99)
        )
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        optimizer = {
            "Adam": optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
            "AdamW": optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
            "SGD": optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay),
            "RMSprop": optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay),
            "Adagrad": optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        }[optimizer_name]

        # Select scheduler
        scheduler = {
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6),
            "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        }[scheduler_name]

        # Train the model
        best_model = train_model(
            model,
            train_loader,
            val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=10,
            accumulation_steps=accumulation_steps,
        )

        # Evaluate the model on the validation set (optimized inference mode)
        with torch.inference_mode():
            metrics = evaluate_model(best_model, val_loader)
        val_f1 = metrics["F1 Score"]  # Extracting F1 Score for optimization

        print(f"🔍 Fold {fold+1}/{kfold} - F1 Score: {val_f1:.4f}")

        # Save the best fold
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_fold = fold
            best_train_indices = train_indices
            best_val_indices = val_indices

    # Save the best fold's indices for reproducibility
    torch.save(best_train_indices, os.path.join(save_dir, "best_train_indices.pt"))
    torch.save(best_val_indices, os.path.join(save_dir, "best_val_indices.pt"))

    print(f"✅ Best Fold: {best_fold+1}/{kfold} saved in {save_dir}")

    return best_f1  # Optuna will maximize this score
