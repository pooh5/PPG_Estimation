# TPP Estimation using ResNet Model

import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, r2_score

import ipywidgets as widgets
import tk_utils
import training_header

# Set working directory
os.chdir('/home/pooja/tpp_ppg_training/')

# Reload training utilities
importlib.reload(training_header)
from training_header import *

# Set torch multiprocessing strategy
torch.multiprocessing.set_sharing_strategy('file_system')
print("Sharing strategy:", torch.multiprocessing.get_sharing_strategy())

# Create folder for figures
fig_folder = "./figures_resnet"
os.makedirs(fig_folder, exist_ok=True)

# Build the ResNet model

def get_model_v3(num_discrete, verbose=False):
    model = MyFirstResNet(in_channels=1, num_discrete=num_discrete)

    if verbose:
        print(model)
        layer_count = sum(1 for _ in model.modules())
        print(f"Total number of layers: {layer_count}")

        kernel_sizes = [15] * 5
        strides = [4] * 5
        receptive_field = 1
        for k, s in zip(kernel_sizes, strides):
            receptive_field += (k - 1) * s

        print(f"Estimated Receptive Field: {receptive_field}")

    return model

# Create and test a sample model
model = get_model_v3(2, verbose=True)
print(f"Model has {sum(np.prod(p.size()) for p in model.parameters()) / 1e6:.2f}M parameters")

with torch.no_grad():
    model.to(DEVICE)
    test_x = torch.tensor(np.random.randn(256, 1, 3600), dtype=torch.float32, device=DEVICE)
    test_w = [torch.tensor(np.random.randn(256), dtype=torch.float32, device=DEVICE) for _ in range(2)]
    test_y = model(test_x, *test_w)
    print(f"Input shape: {test_x.shape}, Output shape: {test_y.shape}")
    model.cpu()

# Load dataset index files

index_file = "train_dataset_index_r2_gt_5_and_pcrit_gt_0_pcrit_lt_map_pp_gt_10_map_lt_120_pcrit_lt_100.pkl"
train_dataset_index = load_dataset_index(index_file)
tune_dataset_index = load_dataset_index(index_file)
test_dataset_index = load_dataset_index(index_file)

logger.info("Resnet, TPP: M1.5: weight_decay=1e-01, l1_reg=1e-04")

# Load datasets
keys_to_load = ["ppg", "map", "dbp", "hr", "tpp"]
train_dset = TPP_PPG_dataset(train_dataset_index, "segment", keys_to_load, percentage=100)
tune_dset = TPP_PPG_dataset(tune_dataset_index, "segment", keys_to_load, percentage=100)

# Set up data loaders
train_loader = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=8, drop_last=True, prefetch_factor=2)
tune_loader = DataLoader(tune_dset, batch_size=256, shuffle=True, num_workers=8, drop_last=True, prefetch_factor=2)

# Train the model
model = get_model_v3(num_discrete=len(keys_to_load)-2)

model, trajectory = training_loop(
    model, train_loader, tune_loader,
    total_epochs=10, lr=1e-3,
    weight_decay=1e-01, l1_reg=1e-04,
    early_stopping=-1,
    scheduler_and_args=[StepLR, {"step_size": 3, "gamma": 0.2}],
    warmup_epochs=0,
    gradient_accumulation=1,
    criterion="mse",
    batch_cosine=False,
    verbose=True,
    checkpoints_name="M1.5.ckpt"
)

# Save model and training logs
os.makedirs("runs", exist_ok=True)
torch.save(model, "runs/M1.5.pth")
torch.save(trajectory, "runs/M1.5_trajectory.pth")

# Evaluate the model
model.to(DEVICE)
model.eval()
tune_labels, tune_preds = [], []

with torch.no_grad():
    for batch_data in tune_loader:
        ppg = batch_data[0].to(DEVICE)
        discretes = [d.to(DEVICE) for d in batch_data[1:-1]]
        label = batch_data[-1]

        tune_labels.append(label.cpu().numpy())
        tune_preds.append(model(ppg, *discretes).squeeze().cpu().numpy())

# Combine predictions and calculate metrics
tune_labels = np.concatenate(tune_labels)
tune_preds = np.concatenate(tune_preds)

rmse = np.sqrt(mean_squared_error(tune_labels, tune_preds))
mead = np.median(np.abs(tune_labels - tune_preds))
mae = np.mean(np.abs(tune_labels - tune_preds))
r2 = r2_score(tune_labels, tune_preds)

# Plot predictions vs true values
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
h = ax.hist2d(tune_labels, tune_preds, bins=np.linspace(0, 100, 200), cmap='Blues')
plt.colorbar(h[3], ax=ax)
plt.xlabel('True Labels')
plt.ylabel('Predictions')
plt.annotate(f'RMSE: {rmse:.2f} mmHg\nMAE: {mae:.2f} mmHg\nMeAD: {mead:.2f} mmHg\nR²: {r2:.2f}',
             xy=(0.95, 0.05), xycoords='axes fraction', fontsize=12, ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
plt.savefig(f'{fig_folder}/histogram_plot.png')
plt.close(fig)

# Plot training loss and learning rate
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
axs[0].plot(trajectory["epoch_train_loss"], label="Train Loss")
axs[0].plot(trajectory["epoch_val_loss"], label="Tune Loss")
axs[0].set(xlabel="Epoch", ylabel="Loss")
axs[0].legend()

axs[1].plot(trajectory["lr"])
axs[1].set(xlabel="Epoch", ylabel="Learning Rate")

plt.savefig(f'{fig_folder}/loss_lr_plot.png')
plt.close(fig)
