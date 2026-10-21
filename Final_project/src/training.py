import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.utils import set_seed


def fit(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    patience=10,
    seed=42,
    device=None,
    **kwargs
):
    # fix seed for reproducible training runs
    set_seed(seed)

    # select device automatically if not forced
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    # convert numpy inputs or clone existing tensors to avoid reference issues
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    else:
        X_train_tensor = X_train.clone().detach().to(torch.float32)
        y_train_tensor = y_train.clone().detach().to(torch.long)

    if isinstance(X_val, np.ndarray):
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    else:
        X_val_tensor = X_val.clone().detach().to(torch.float32)
        y_val_tensor = y_val.clone().detach().to(torch.long)

    # prepare dataloaders (shuffle train only, keep val fixed)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # cross entropy expects raw logits from the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = None
    epochs_completed = 0

    t_start = time.time()

    for epoch in range(epochs):
        # --------------------------------------------------
        # training loop
        # --------------------------------------------------
        model.train()  # enable dropout / batchnorm
        
        running_train_loss = 0.0
        train_correct_count = 0
        train_total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # zero out accumulated gradients
            optimizer.zero_grad()
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # backprop and update weights
            loss.backward()
            optimizer.step()

            # accumulate metrics multiplied by batch size for exact averages later
            running_train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            
            train_total_samples += batch_y.size(0)
            train_correct_count += (preds == batch_y).sum().item()

        epoch_train_loss = running_train_loss / train_total_samples
        epoch_train_acc = train_correct_count / train_total_samples

        # --------------------------------------------------
        # validation loop
        # --------------------------------------------------
        model.eval()  # turn off dropout
        
        running_val_loss = 0.0
        val_correct_count = 0
        val_total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                running_val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                
                val_total_samples += batch_y.size(0)
                val_correct_count += (preds == batch_y).sum().item()

        epoch_val_loss = running_val_loss / val_total_samples
        epoch_val_acc = val_correct_count / val_total_samples

        # record history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_accuracy"].append(epoch_train_acc)
        history["val_accuracy"].append(epoch_val_acc)

        epochs_completed += 1

        # --------------------------------------------------
        # early stopping check
        # --------------------------------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # save best weights copy in memory
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # restore best weights before exiting early
                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                break

    # if finished all epochs without triggering early stop, restore best model anyway
    if best_model_weights is not None and patience_counter < patience:
        model.load_state_dict(best_model_weights)

    total_time = time.time() - t_start

    return {
        "model": model,
        "history": history,
        "training_time": total_time,
        "epochs_completed": epochs_completed
    }