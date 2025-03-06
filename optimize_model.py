import os
import torch
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from classifier.data import split_dataset
from classifier.optimize import objective


if __name__ == "__main__":
    dataset_dir = "WildShapesDataset/images"
    if not os.path.exists("dataset_splits"):
        split_dataset(dataset_dir)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # Run 30 trials

    # Save the best hyperparameters
    best_params = study.best_params
    torch.save(best_params, "best_hyperparameters.pt")

    print("\n🚀 Best Hyperparameters Found:", best_params)

    # Convert all saved histories into a single file
    all_data = pd.concat([pd.read_csv(f) for f in os.listdir() if f.startswith("trial_") and f.endswith("_history.csv")])
    all_data.to_csv("optimization_history.csv", index=False)
    print("📊 Full optimization history saved to 'optimization_history.csv'.")

    # Plot Optimization Results
    def plot_optimization_results(df):
        """Generates plots for the optimization process."""
        plt.figure(figsize=(12, 6))

        # Scatter plot of F1 Score vs. Learning Rate
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x="Learning Rate", y="F1 Score", hue="Optimizer", palette="deep", alpha=0.8)
        plt.xscale("log")
        plt.title("F1 Score vs. Learning Rate")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("F1 Score")
        plt.legend()

        # Scatter plot of F1 Score vs. Batch Size
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x="Batch Size", y="F1 Score", palette="coolwarm")
        plt.title("F1 Score vs. Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("F1 Score")

        plt.tight_layout()
        plt.show()

    # Load final optimization history and generate plots
    final_df = pd.read_csv("optimization_history.csv")
    plot_optimization_results(final_df)
