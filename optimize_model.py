import os
import torch
import optuna
from classifier.optimize import objective
from classifier.data import split_dataset, set_seed

if __name__ == "__main__":
    # Definindo a seed para todos os processos
    set_seed(2108)

    dataset_dir = "WildShapesDataset/images"
    if not os.path.exists("dataset_splits"):
        split_dataset(dataset_dir)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Run 50 trials

    # Save the best hyperparameters
    best_params = study.best_params
    torch.save(best_params, "best_hyperparameters.pt")

    print("\n🚀 Best Hyperparameters Found:", best_params)
