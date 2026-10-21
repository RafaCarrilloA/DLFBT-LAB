import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. EVALUATE MODEL & COMPUTE METRICS
# ==========================================
def evaluate_model(model, X, y, batch_size=64, device=None, **kwargs):
    # setup target device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    model = model.to(device)
    model.eval()  # turn off dropout and freeze batchnorm

    # make sure inputs are proper torch tensors with correct types
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.clone().detach().to(torch.float32)

    if isinstance(y, np.ndarray):
        y_tensor = torch.tensor(y, dtype=torch.long)
    else:
        y_tensor = y.clone().detach().to(torch.long)

    # create loader so we don't blow up GPU memory on large test sets
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_pred_list = []
    y_true_list = []

    # evaluation loop (no autograd needed here)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            
            logits = model(batch_x)
            # get class index with highest score
            preds = torch.argmax(logits, dim=1)
            
            # send back to cpu to store as numpy
            y_pred_list.extend(preds.cpu().numpy())
            y_true_list.extend(batch_y.numpy())

    preds_arr = np.array(y_pred_list)
    targets_arr = np.array(y_true_list)

    # compute standard classification metrics
    acc = accuracy_score(targets_arr, preds_arr)
    macro_prec = precision_score(targets_arr, preds_arr, average='macro', zero_division=0)
    macro_rec = recall_score(targets_arr, preds_arr, average='macro', zero_division=0)
    macro_f1_val = f1_score(targets_arr, preds_arr, average='macro', zero_division=0)
    weighted_f1_val = f1_score(targets_arr, preds_arr, average='weighted', zero_division=0)

    # dict matching project spec requirements
    metrics = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1_val,
        "weighted_f1": weighted_f1_val,
        "predictions": preds_arr
    }
    
    return metrics


# ==========================================
# 2. COUNT TRAINABLE PARAMETERS
# ==========================================
def count_parameters(model, **kwargs):
    # loop over parameter tensors and count only trainable weights
    total_trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_trainable_params += param.numel()
            
    return total_trainable_params


# ==========================================
# 3. MEASURE INFERENCE TIME
# ==========================================
def measure_inference_time(model, X, device=None, **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    model = model.to(device)
    model.eval()

    # prepare input matrix on target device
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    else:
        X_tensor = X.clone().detach().to(torch.float32).to(device)

    # synchronize cuda before starting clock if on GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_start = time.time()

    with torch.no_grad():
        _ = model(X_tensor)

    # sync again so we actually measure kernel execution time, not just dispatch
    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed_time = time.time() - t_start

    return elapsed_time