import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import FlexibleCompensator
from fcs_preprocessing import preprocess_fcs_for_model
from loss_functions import PopulationLoss
from fcsparser import parse
import numpy as np
import logging

# --- Logging setup ---
logging.basicConfig(
    filename='training.log',
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# --- Data Preparation ---
class FCSPairDataset(Dataset):
    def __init__(self, files_dir):
        self.X = []
        self.Y = []
        not_unmixed_cols_all = []
        unmixed_cols_all = []
        self.files = []
        for fname in os.listdir(files_dir):
            if not fname.lower().endswith('.fcs') or '_unmixed' in fname or '_not_unmixed' in fname:
                continue
            try:
                meta, df = parse(os.path.join(files_dir, fname), reformat_meta=True)
            except Exception as e:
                logging.warning(f"[ERROR] {fname}: {e}")
                continue
            unmixed_cols = [col for col in df.columns if 'spectral' in col.lower()]
            not_unmixed_cols = [col for col in df.columns if 'spectral' not in col.lower()]
            if not unmixed_cols or not not_unmixed_cols:
                continue
            # Ensure columns are sorted and aligned by channel name (after removing known suffixes/prefixes)
            def clean_col(col):
                return col.lower().replace('-a', '').replace('-comp', '').replace(' ', '').replace('_', '')
            unmixed_cols_clean = sorted(unmixed_cols, key=clean_col)
            not_unmixed_cols_clean = sorted(not_unmixed_cols, key=clean_col)
            unmixed_cols_all.append(set(unmixed_cols_clean))
            not_unmixed_cols_all.append(set(not_unmixed_cols_clean))
            self.files.append(fname)
        if not unmixed_cols_all or not not_unmixed_cols_all:
            raise RuntimeError("No valid FCS files with both unmixed and not-unmixed data found.")
        # Use intersection and sort by cleaned name for alignment
        def sort_cols(cols):
            return sorted(cols, key=lambda c: c.lower().replace('-a', '').replace('-comp', '').replace(' ', '').replace('_', ''))
        self.common_unmixed_cols = sort_cols(set.intersection(*unmixed_cols_all))
        self.common_not_unmixed_cols = sort_cols(set.intersection(*not_unmixed_cols_all))
        # Load all data
        for fname in self.files:
            meta, df = parse(os.path.join(files_dir, fname), reformat_meta=True)
            unmixed_df = df[self.common_unmixed_cols]
            not_unmixed_df = df[self.common_not_unmixed_cols]
            min_len = min(len(unmixed_df), len(not_unmixed_df))
            # Remove NaNs
            x = not_unmixed_df.iloc[:min_len].to_numpy()
            y = unmixed_df.iloc[:min_len].to_numpy()
            mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
            x = x[mask]
            y = y[mask]
            self.X.append(x)
            self.Y.append(y)
        self.X = np.concatenate(self.X)
        self.Y = np.concatenate(self.Y)
        # Normalize
        self.x_mean = self.X.mean(axis=0)
        self.x_std = self.X.std(axis=0) + 1e-8
        self.y_mean = self.Y.mean(axis=0)
        self.y_std = self.Y.std(axis=0) + 1e-8
        self.X = (self.X - self.x_mean) / self.x_std
        self.Y = (self.Y - self.y_mean) / self.y_std
        logging.info(f"[DATASET] X shape: {self.X.shape}, Y shape: {self.Y.shape}")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# --- Training ---
def train():
    files_dir = "files"
    batch_size = 1024  # smaller batch size for stability
    epochs = 200       # more epochs for better convergence
    patience = 20      # more patience for early stopping
    dataset = FCSPairDataset(files_dir)
    in_dim = len(dataset.common_not_unmixed_cols)
    out_dim = len(dataset.common_unmixed_cols)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[DEVICE] Using device: {device}")
    model = FlexibleCompensator(in_dim, [1024, 2048, 1024, 512], out_dim, activation="SiLU", norm="LayerNorm", dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = PopulationLoss().to(device)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    pred = model(Xb)
                    loss = loss_fn(pred, Yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(Xb)
                loss = loss_fn(pred, Yb)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        pred = model(Xb)
                        loss = loss_fn(pred, Yb)
                else:
                    pred = model(Xb)
                    loss = loss_fn(pred, Yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        logging.info(f"Epoch {epoch}: Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f} | LR {optimizer.param_groups[0]['lr']:.2e}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "trained_model.pt")
            logging.info("[MODEL] Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("[EARLY STOP] No improvement, stopping training.")
                break
    # After training, visualize predictions vs. targets
    import matplotlib.pyplot as plt
    model.eval()
    X_sample, Y_sample = next(iter(val_loader))
    X_sample, Y_sample = X_sample.to(device), Y_sample.to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                pred = model(X_sample)
        else:
            pred = model(X_sample)
    pred = pred.cpu().numpy()
    Y_sample = Y_sample.cpu().numpy()
    # Plot first 5 output channels
    for i in range(min(5, pred.shape[1])):
        plt.figure()
        plt.scatter(Y_sample[:, i], pred[:, i], alpha=0.2)
        plt.xlabel('True (normalized)')
        plt.ylabel('Predicted (normalized)')
        plt.title(f'Output Channel {i+1}')
        plt.plot([-3, 3], [-3, 3], 'r--')
        plt.savefig(f'prediction_vs_true_channel_{i+1}.png')
        plt.close()
    logging.info("[PLOT] Saved prediction vs. true plots for first 5 output channels.")
    # Save the columns used for training for later inference
    import json
    with open("model_columns.json", "w") as f:
        json.dump({
            "not_unmixed_cols": dataset.common_not_unmixed_cols,
            "unmixed_cols": dataset.common_unmixed_cols
        }, f)

if __name__ == "__main__":
    if os.path.exists("trained_model.pt"):
        os.remove("trained_model.pt")
        print("Deleted old trained_model.pt. Please re-run training to generate a new compatible model.")
    if os.path.exists("model_columns.json"):
        os.remove("model_columns.json")
        print("Deleted old model_columns.json.")
    train()
