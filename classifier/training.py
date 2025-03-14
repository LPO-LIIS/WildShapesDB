import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from classifier.model import Shape2DClassifier


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
    use_compile=True,  # Usa torch.compile se disponível
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Usa torch.compile() se disponível
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ torch.compile falhou: {e}")

    # Função de perda
    criterion = nn.CrossEntropyLoss()

    # Habilita Mixed Precision Training apenas se houver GPU
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Controle de early stopping e checkpoint
    best_val_f1 = 0.0
    no_improve_epochs = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter("logs/tensorboard")

    # Loop de treinamento
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

            # Usa autocast apenas se AMP estiver habilitado
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Gradient Accumulation
            scaler.scale(loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                # Verifica se há NaNs antes de continuar
                if not torch.isfinite(loss).all():
                    optimizer.zero_grad()
                    continue

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

            # Armazena as previsões e rótulos diretamente como tensores
            _, predicted = outputs.max(1)
            all_train_preds.append(predicted)
            all_train_labels.append(labels)

            if batch_idx % 10 == 0:  # Atualiza tqdm a cada 10 batches
                train_loader_tqdm.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)  # Normalização

        # Concatena todas as previsões e rótulos como tensores na GPU
        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)

        # Move para CPU apenas na hora de calcular métricas
        train_f1 = f1_score(all_train_labels.cpu(), all_train_preds.cpu(), average="weighted")

        # Fase de Validação
        model.eval()
        with torch.no_grad():  # Garante que não haverá cálculos de gradiente
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Usa autocast apenas se AMP estiver habilitado
                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)  # Normalização
        val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")

        if epoch > 0:
            scheduler.step()

        # Logging
        log = (
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}\n"
        )

        tqdm.write(log)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # Salva apenas o melhor modelo
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            model_path = os.path.join(checkpoint_dir, f"best_model_f1.pth")
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"🔹 Novo melhor modelo salvo: {model_path} (F1: {best_val_f1:.4f})")

        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            tqdm.write(f"⏹️ Early stopping ativado após {patience} épocas sem melhoria.")
            break

    writer.close()

    # Carregar o melhor modelo salvo
    best_model_path = os.path.join(checkpoint_dir, "best_model_f1.pth")
    best_model = Shape2DClassifier(num_classes=9)

    # Carregar o state_dict salvo
    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)

    # Checa se as chaves têm `_orig_mod` e remove se necessário
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("⚠️ Removendo prefixo '_orig_mod.' do state_dict.")

    # Aplicar ao modelo
    best_model.load_state_dict(state_dict)

    tqdm.write(f"✅ Melhor modelo carregado de: {best_model_path}")

    return best_model
