#!/usr/bin/env python3
"""Evaluate the Siamese Network on the ML Test Split."""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from evaluation.ml.model import SiameseNetwork

log = logging.getLogger("ml_experiment.evaluate_ml")

def evaluate_ml(metric="euclidean"):
    data_dir = _BACKEND / "evaluation" / "ml" / "data"
    test_file = data_dir / "test_data.npy"
    model_path = _BACKEND / "evaluation" / "ml" / f"best_model_{metric}.pth"
    scaler_mean_path = data_dir / f"scaler_mean_{metric}.npy"
    scaler_std_path = data_dir / f"scaler_std_{metric}.npy"
    
    if not test_file.exists() or not model_path.exists():
        log.error("Missing test data or model weights.")
        return

    test_data = np.load(test_file)
    n_users, n_sessions, n_features = test_data.shape
    
    # Load Scaler
    mean = np.load(scaler_mean_path)
    std = np.load(scaler_std_path)
    
    # Standardize Test Data
    test_data = (test_data - mean) / std
    test_data = np.nan_to_num(test_data, nan=0.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(input_dim=n_features, embedding_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Get total parameters (model size)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Compute embeddings for all test data
    log.info(f"Computing embeddings for {n_users} users...")
    start_time = time.time()
    
    with torch.no_grad():
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
        # shape: (N_users, N_sessions, N_features) -> flatten
        flat_tensor = test_tensor.view(-1, n_features)
        embeddings = model.get_embedding(flat_tensor)
        embeddings = embeddings.view(n_users, n_sessions, -1).cpu().numpy()
        
    inference_time = (time.time() - start_time) / (n_users * n_sessions)
    
    genuine_scores = []
    impostor_scores = []
    
    n_impostors_per_user = 100
    
    log.info("Scoring attempts...")
    
    for u in range(n_users):
        # Enrollment: Sessions 0-7. Take the mean embedding.
        enroll_emb = np.mean(embeddings[u, :8, :], axis=0)
        
        # Genuine: Sessions 8, 9
        for s in (8, 9):
            attempt_emb = embeddings[u, s, :]
            
            if metric == "euclidean":
                dist = np.linalg.norm(enroll_emb - attempt_emb)
            else: # cosine
                num = np.dot(enroll_emb, attempt_emb)
                den = np.linalg.norm(enroll_emb) * np.linalg.norm(attempt_emb)
                dist = 1.0 - (num / den if den > 0 else 0.0)
                
            genuine_scores.append(dist)
            
        # Impostor: Session 8 of other random users
        impostor_indices = np.random.choice([i for i in range(n_users) if i != u], size=n_impostors_per_user, replace=False)
        for imp_u in impostor_indices:
            attempt_emb = embeddings[imp_u, 8, :]
            if metric == "euclidean":
                dist = np.linalg.norm(enroll_emb - attempt_emb)
            else: # cosine
                num = np.dot(enroll_emb, attempt_emb)
                den = np.linalg.norm(enroll_emb) * np.linalg.norm(attempt_emb)
                dist = 1.0 - (num / den if den > 0 else 0.0)
                
            impostor_scores.append(dist)

    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_scores = np.concatenate([-genuine, -impostor])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = -thresholds[eer_idx]
    
    log.info(f"--- ML RESULTS ({metric.upper()}) ---")
    log.info(f"ROC-AUC: {roc_auc:.4f}")
    log.info(f"EER:     {eer:.4f} (at threshold {eer_threshold:.4f})")
    log.info(f"Inference Latency: {inference_time*1000:.4f} ms / session")
    log.info(f"Model Size: {total_params} parameters")
    
    target_fars = [0.01, 0.001]
    for tfar in target_fars:
        idx = np.where(fpr <= tfar)[0][-1]
        log.info(f"At FAR <= {tfar:.3f}: FRR = {fnr[idx]:.4f} (threshold {-thresholds[idx]:.4f})")
        
    return {
        "auc": roc_auc,
        "eer": eer,
        "inference_ms": inference_time * 1000,
        "model_params": total_params
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    torch.manual_seed(42)
    
    evaluate_ml(metric="euclidean")
    evaluate_ml(metric="cosine")
