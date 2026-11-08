"""HISTORICAL SOURCE SNAPSHOT; NOT THE EXACT GENERATING RUN.

WARNING: executes training and writes outputs on import. Do not import for audit.
The unsupported Mamba(n_layers=...) argument and last-timestep selection (rather
than the report's temporal max pooling) are deliberately NOT repaired here.
Only the input path and this notice were changed. See ../README.md.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
import json

RAW_PATH = os.environ.get('GAIT_DATA_DIR', 'data/raw')

CHANNELS = [3, 6, 7, 9, 11, 14, 15, 17, 18, 19]  # customizable

MODEL_SAVE_DIR = "outputs/models"
PLOTS_SAVE_DIR = "outputs/plots"
METRICS_SAVE_DIR = "outputs/metrics"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Load files & subject-level split
# --------------------------------------------------

def find_all_txt_files(root_dir):
    txt_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


def get_subject_id(file_path):
    return os.path.basename(file_path)[:6]   # GaCo01, GaPt02, ...


print("\nScanning dataset...")
all_files = find_all_txt_files(RAW_PATH)

print("Found:", len(all_files))

valid_files = []
for file_path in all_files:
    try:
        d = np.loadtxt(file_path)
        if d.ndim == 2 and d.shape[1] >= 20:
            valid_files.append(file_path)
        else:
            print("Skipping:", file_path, "shape:", d.shape)
    except:
        print("Read error:", file_path)

all_files = valid_files
print("Valid files:", len(all_files))

subjects = np.array([get_subject_id(f) for f in all_files])
unique_subjects = np.unique(subjects)

# Label extraction: 1=Parkinson, 0=Control
subject_labels = np.array([1 if s[2:4] == "Pt" else 0 for s in unique_subjects])

train_subj, val_subj = train_test_split(
    unique_subjects,
    test_size=0.20,
    random_state=42,
    stratify=subject_labels
)

train_files = [f for f in all_files if get_subject_id(f) in train_subj]
val_files   = [f for f in all_files if get_subject_id(f) in val_subj]

print("Train files:", len(train_files))
print("Val files:", len(val_files))


# --------------------------------------------------
# 2. Dataset class
# --------------------------------------------------

class GaitDataset(Dataset):
    def __init__(self, file_list, channels):
        self.files = file_list
        self.channels = channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        raw = np.loadtxt(f)

        # X: selected columns
        X = raw[:, self.channels].astype(np.float32)

        # label from filename
        subj = get_subject_id(f)
        y = 1 if subj[2:4] == "Pt" else 0

        # reshape to (sequence_length, channels)
        return torch.tensor(X), torch.tensor(y, dtype=torch.long)


train_ds = GaitDataset(train_files, CHANNELS)
val_ds   = GaitDataset(val_files, CHANNELS)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)


# --------------------------------------------------
# 3. Mamba model for time-series classification
# --------------------------------------------------

class MambaClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=2,
        d_model=128,
        n_layers=4,
        expand=2,
        dt_rank=32,
        d_state=32
    ):
        super().__init__()

        self.project_in = nn.Linear(input_dim, d_model)

        self.mamba = Mamba(
            d_model=d_model,
            n_layers=n_layers,
            expand=expand,
            dt_rank=dt_rank,
            d_state=d_state
        )

        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.project_in(x)
        y = self.mamba(x)
        y = y[:, -1, :]   # last timestep
        return self.cls_head(y)



model = MambaClassifier(input_dim=len(CHANNELS)).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    verbose=True
)

EARLY_STOPPING_PATIENCE = 10
best_val_loss = float("inf")
early_stop_counter = 0

# --------------------------------------------------
# 4. Training Loop
# --------------------------------------------------

EPOCHS = 50
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()
    running_loss, correct, total = 0, 0, 0

    for X, y in train_loader:
        X, y = X.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validation phase ---
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.cuda(), y.cuda()
            out = model(X)
            loss = criterion(out, y)

            running_loss += loss.item() * X.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch} | "
          f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

    # Save checkpoint for this epoch
    torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/checkpoint_epoch{epoch}.pth")

    # LR Scheduler
    scheduler.step(val_loss)

    # --- Early Stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0

        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/best_model.pth")
        print("Updated best model.")
    else:
        early_stop_counter += 1
        print(f"No improvement. EarlyStop counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered!")
        break



# --------------------------------------------------
# 5. Save final full model
# --------------------------------------------------
torch.save(model, f"{MODEL_SAVE_DIR}/mamba_classifier_full.pth")
print("\nFull model saved.")


# --------------------------------------------------
# 6. Save curves
# --------------------------------------------------
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{PLOTS_SAVE_DIR}/loss_curve.png")
plt.close()

plt.figure()
plt.plot(train_accs, label="train")
plt.plot(val_accs, label="val")
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(f"{PLOTS_SAVE_DIR}/accuracy_curve.png")
plt.close()

print("Training curves saved.")


# --------------------------------------------------
# 7. Classification report on validation set
# --------------------------------------------------

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X, y in val_loader:
        X = X.cuda()
        out = model(X)
        _, pred = torch.max(out, 1)
        y_true.extend(y.tolist())
        y_pred.extend(pred.cpu().tolist())

report = classification_report(
    y_true, y_pred,
    target_names=["Control", "Parkinson"],
    output_dict=True
)

print("\nValidation Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Control", "Parkinson"]))

with open(f"{METRICS_SAVE_DIR}/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Classification report saved.")
