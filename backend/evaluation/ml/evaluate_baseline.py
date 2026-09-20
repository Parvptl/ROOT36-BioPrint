#!/usr/bin/env python3
"""Evaluate the BioPrint Phase 2 baseline on the ML Test Split.

Protocol:
- For each test user:
    - Enrollment: Sessions 0-7 (8 sessions) -> build_calibrated_profile
    - Genuine Attempts: Sessions 8, 9
- For impostor scores:
    - Impostor Attempts: Session 8 from a subset of OTHER test users

This identical protocol will be used to evaluate the Siamese ML model.
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.fingerprint.calibration import build_calibrated_profile
from app.behavioral.fingerprint.population import PopulationPrior
from app.behavioral.fingerprint.scoring import score_identity
from app.behavioral.fingerprint.profile import BehaviorProfile

log = logging.getLogger("ml_experiment.evaluate_baseline")

def evaluate_baseline():
    data_dir = _BACKEND / "evaluation" / "ml" / "data"
    test_file = data_dir / "test_data.npy"
    features_file = data_dir / "features.json"
    
    if not test_file.exists():
        log.error("Test data not found. Run generate_ml_dataset.py first.")
        return

    test_data = np.load(test_file)  # (N_users, N_sessions, N_features)
    with open(features_file, "r") as f:
        feature_names = json.load(f)

    n_users, n_sessions, n_features = test_data.shape
    log.info(f"Loaded Test Data: {n_users} users, {n_sessions} sessions, {n_features} features")

    prior = PopulationPrior()
    
    genuine_scores = []
    impostor_scores = []
    
    n_impostors_per_user = 100
    
    profiles: list[BehaviorProfile] = []
    
    log.info("Building baseline profiles (8 sessions)...")
    for u in range(n_users):
        sessions = []
        for s in range(8):
            sess_dict = {}
            for f_idx, f_name in enumerate(feature_names):
                val = test_data[u, s, f_idx]
                if np.isfinite(val):
                    sess_dict[f_name] = float(val)
            sessions.append(sess_dict)
            
        profile = build_calibrated_profile(sessions, prior, [])
        profiles.append(profile)

    log.info("Scoring attempts...")
    for u in range(n_users):
        profile = profiles[u]
        
        for s in (8, 9):
            attempt = {}
            for f_idx, f_name in enumerate(feature_names):
                val = test_data[u, s, f_idx]
                if np.isfinite(val):
                    attempt[f_name] = float(val)
                    
            res = score_identity(profile, attempt)
            if res.is_comparable:
                genuine_scores.append(res.score)
                
        impostor_indices = np.random.choice([i for i in range(n_users) if i != u], size=n_impostors_per_user, replace=False)
        for imp_u in impostor_indices:
            attempt = {}
            for f_idx, f_name in enumerate(feature_names):
                val = test_data[imp_u, 8, f_idx]
                if np.isfinite(val):
                    attempt[f_name] = float(val)
            
            res = score_identity(profile, attempt)
            if res.is_comparable:
                impostor_scores.append(res.score)

    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    
    log.info(f"Genuine scores: {len(genuine)}")
    log.info(f"Impostor scores: {len(impostor)}")
    
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_scores = np.concatenate([-genuine, -impostor])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = -thresholds[eer_idx]
    
    log.info(f"--- BASELINE RESULTS ---")
    log.info(f"ROC-AUC: {roc_auc:.4f}")
    log.info(f"EER:     {eer:.4f} (at threshold {eer_threshold:.4f})")
    
    log.info(f"Genuine p50: {np.median(genuine):.4f}")
    log.info(f"Impostor p50: {np.median(impostor):.4f}")
    
    target_fars = [0.01, 0.001]
    for tfar in target_fars:
        idx = np.where(fpr <= tfar)[0][-1]
        log.info(f"At FAR <= {tfar:.3f}: FRR = {fnr[idx]:.4f} (threshold {-thresholds[idx]:.4f})")
        
    return {
        "auc": roc_auc,
        "eer": eer,
        "genuine_scores": genuine.tolist(),
        "impostor_scores": impostor.tolist(),
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    evaluate_baseline()
