#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from evaluation.ml.model import SiameseNetwork, ContrastiveLoss

log = logging.getLogger("ml_experiment.train")

class SiameseDataset(Dataset):
    def __init__(self, data_npy: Path, n_pairs_per_user: int = 10):
        super().__init__()
        # Load dataset of shape (N_users, N_sessions, N_features)
        self.data = np.load(data_npy)
        self.n_users, self.n_sessions, self.n_features = self.data.shape
        self.n_pairs_per_user = n_pairs_per_user
        
        # Calculate feature medians and stds globally for standardization
        # Flatten all sessions of all users
        flat = self.data.reshape(-1, self.n_features)
        
        # We need to compute statistics ignoring NaNs
        self.mean = np.nanmean(flat, axis=0)
        self.std = np.nanstd(flat, axis=0)
        self.std[self.std == 0] = 1.0 # Prevent division by zero
        
        # Standardize data in place
        self.data = (self.data - self.mean) / self.std
        
        self._generate_pairs()
        
    def _generate_pairs(self):
        self.pairs = []
        self.labels = []
        
        log.info(f"Generating {self.n_pairs_per_user} pos/neg pairs per user...")
        
        for u in range(self.n_users):
            # Positive pairs
            for _ in range(self.n_pairs_per_user):
                s1, s2 = np.random.choice(self.n_sessions, 2, replace=False)
                self.pairs.append((self.data[u, s1], self.data[u, s2]))
                self.labels.append(1) # Same user
                
            # Negative pairs
            for _ in range(self.n_pairs_per_user):
                s1 = np.random.choice(self.n_sessions)
                u2 = np.random.choice([i for i in range(self.n_users) if i != u])
                s2 = np.random.choice(self.n_sessions)
                self.pairs.append((self.data[u, s1], self.data[u2, s2]))
                self.labels.append(0) # Different user

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        y = self.labels[idx]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


def compute_eer(distances, labels):
    from sklearn.metrics import roc_curve
    # distances are lower for same user. We need scores higher for same user for roc_curve
    fpr, tpr, _ = roc_curve(labels, -distances)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[eer_idx] + fnr[eer_idx]) / 2

def train_model(metric="euclidean", margin=2.0, epochs=20, batch_size=128):
    data_dir = _BACKEND / "evaluation" / "ml" / "data"
    train_file = data_dir / "train_data.npy"
    val_file = data_dir / "val_data.npy"
    
    if not train_file.exists():
        log.error("Run generate_ml_dataset.py first.")
        return
        
    train_dataset = SiameseDataset(train_file, n_pairs_per_user=20) # 20 pos + 20 neg = 40 pairs/user * 3000 = 120,000
    val_dataset = SiameseDataset(val_file, n_pairs_per_user=10)     # 20 pairs/user * 1000 = 20,000
    
    # Save the scaler parameters (mean, std) so the test set can use them
    np.save(data_dir / f"scaler_mean_{metric}.npy", train_dataset.mean)
    np.save(data_dir / f"scaler_std_{metric}.npy", train_dataset.std)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(input_dim=train_dataset.n_features, embedding_dim=16).to(device)
    
    criterion = ContrastiveLoss(margin=margin, metric=metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_eer = float("inf")
    best_model_path = _BACKEND / "evaluation" / "ml" / f"best_model_{metric}.pth"
    
    import time
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dists = []
        val_labels = []
        
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                out1, out2 = model(x1, x2)
                loss = criterion(out1, out2, y)
                val_loss += loss.item()
                
                if metric == "euclidean":
                    dist = torch.nn.functional.pairwise_distance(out1, out2)
                else:
                    dist = 1 - torch.nn.functional.cosine_similarity(out1, out2)
                    
                val_dists.extend(dist.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_dists = np.array(val_dists)
        val_labels = np.array(val_labels).flatten()
        
        val_eer = compute_eer(val_dists, val_labels)
        
        log.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val EER: {val_eer:.4f}")
        
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  -> Saved new best model (Val EER: {best_val_eer:.4f})")
            
    training_time = time.time() - start_time
    log.info(f"Training completed in {training_time:.2f}s. Best Val EER: {best_val_eer:.4f}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Train both metrics
    log.info("--- Training Euclidean model ---")
    train_model(metric="euclidean", margin=2.0)
    
    log.info("--- Training Cosine model ---")
    train_model(metric="cosine", margin=0.5)
