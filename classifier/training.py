import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import glob
import re

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs=20,
    checkpoint_dir="checkpoints",
    patience=5,
    accumulation_steps=1,  # Gradient Accumulation
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision Training (AMP)
    scaler = GradScaler()

    # Early stopping and checkpoint management
    best_val_f1 = 0.0
    no_improve_epochs = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter("logs/tensorboard")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        all_train_preds, all_train_labels = [], []
        all_val_preds, all_val_labels = [], []

        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True
        )

        for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()  # Corrected position

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Gradient Accumulation
            scaler.scale(loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            if batch_idx % 10 == 0:  # Update tqdm every 10 batches
                train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)  # Normalization
        train_f1 = f1_score(all_train_labels, all_train_preds, average="weighted")
        train_acc = accuracy_score(all_train_labels, all_train_preds)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)  # Normalization
        val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")
        val_acc = accuracy_score(all_val_labels, all_val_preds)

        scheduler.step()

        # Logging
        log = (
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}\n"
        )

        tqdm.write(log)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Acc/Validation", val_acc, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # Save only the top 3 best models
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            model_path = os.path.join(checkpoint_dir, f"best_model_f1.pth")
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"🔹 New best model saved: {model_path} (F1: {best_val_f1:.4f})")

            # Remove older models, keeping only the 3 best
            models = sorted(glob.glob(os.path.join(checkpoint_dir, "best_model_f1_*.pth")), 
                            key=lambda x: float(re.search(r"f1_(\d+\.\d+)", x).group(1)), 
                            reverse=True)
            for old_model in models[3:]:  # Keep only top 3
                os.remove(old_model)

        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            tqdm.write(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

    writer.close()

    # Load the best model
    best_model_path = os.path.join(checkpoint_dir, "best_model_f1.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    tqdm.write(f"✅ Loaded best model from: {best_model_path}")

    return model
