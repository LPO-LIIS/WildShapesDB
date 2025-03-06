import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.profiler import profile, ProfilerActivity
from torchinfo import summary  # Alternative for FLOP counting

def compute_flops(model, sample_input):
    """Estimate FLOPs using torchinfo summary (alternative to fvcore)."""
    return summary(model, input_size=sample_input.shape, verbose=0).total_mult_adds / 1e9  # Convert to GFLOPs

def evaluate_model(model, dataloader):
    """
    Evaluate the classification model on a given dataset.

    Parameters:
    model (torch.nn.Module): Trained model.
    dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            inference_times.append(time.time() - start_time)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute standard classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    # Compute FLOPs using torchinfo
    sample_input, _ = next(iter(dataloader))
    sample_input = sample_input.to(device)
    gflops = compute_flops(model, sample_input)

    # Compute inferences per second
    avg_inference_time = np.mean(inference_times)
    inferences_per_second = 1.0 / avg_inference_time if avg_inference_time > 0 else float('inf')

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "GFLOPs": gflops,
        "Inferences per second": inferences_per_second,
        "Confusion Matrix": conf_matrix,
        "Classification Report": class_report
    }

    return metrics
