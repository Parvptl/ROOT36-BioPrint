#!/usr/bin/env python3
"""Generate a synthetic dataset for the Offline ML Feasibility Experiment.

This creates completely disjoint users for Train (3000), Validation (1000), 
and Test (1000) splits using the TypingStyle synthetic generator.

We only extract the 16 KEYBOARD features to ensure fairness with Aalto 
capabilities, even though this is synthetic data.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.behavioral.events import build_session_view
from app.behavioral.features import keyboard
from app.behavioral.features.registry import Modality, specs_for
from tests.factories import TypingStyle, human_session

log = logging.getLogger("ml_experiment.dataset")

# Use a fixed list of 16 keyboard features
KEYBOARD_FEATURES = [spec.name for spec in specs_for(Modality.KEYBOARD)]

_PHRASES = [
    "the quick brown fox jumps over the lazy dog near Sunny fields",
    "seven Wild elephants Danced across the bright meadow at noon",
    "clever foxes Navigate through Hidden valleys and steep Mountains",
    "ancient rivers Flow beneath the Towering marble bridges Quietly",
    "purple Clouds drifted above the Sleeping coastal village today",
    "bright Yellow butterflies Explored the garden throughout warm spring",
    "small Children played Beyond the wooden fence until Evening came",
    "distant Thunder echoed Through the misty mountain Valley below",
    "golden Autumn leaves Scattered across the empty cobblestone road",
    "Fresh morning Dew covered Every blade of grass in Silent meadows",
    "curious Rabbits hopped Beside the crystal clear Stream at dawn",
    "Tall pine Trees swayed Gently in the cool evening breeze tonight",
]

def _make_typing_style(rng: np.random.RandomState) -> dict:
    return {
        "iki_mean_ms": float(rng.uniform(80, 350)),
        "iki_jitter": float(rng.uniform(0.15, 0.55)),
        "dwell_mean_ms": float(rng.uniform(50, 170)),
        "dwell_jitter": float(rng.uniform(0.12, 0.45)),
        "overlap_prob": float(rng.uniform(0.0, 0.50)),
        "pause_prob": float(rng.uniform(0.0, 0.12)),
        "pause_ms": float(rng.uniform(400, 1200)),
        "backspace_prob": float(rng.uniform(0.0, 0.10)),
        "right_shift_prob": float(rng.choice([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])),
        "tab_between_fields": bool(rng.random() > 0.3),
        "focus_delay_ms": float(rng.uniform(150, 600)),
        "presubmit_ms": float(rng.uniform(300, 1200)),
        "pointer_speed": float(rng.uniform(0.5, 3.0)),
        "pointer_wobble": float(rng.uniform(0.8, 4.0)),
    }

def _generate_session(phrase: str, style_params: dict, seed: int):
    style = TypingStyle(**style_params)
    session = human_session(
        phrase=phrase,
        nonce=f"ml-experiment-{seed:08d}",
        style=style,
        seed=seed,
        use_pointer=False,
    )
    view = build_session_view(session)
    raw = keyboard.extract(view)

    features = {}
    for name in KEYBOARD_FEATURES:
        if name in raw:
            value, n_obs = raw[name]
            features[name] = value if np.isfinite(value) else np.nan
        else:
            features[name] = np.nan
    return features

def generate_split(name: str, n_users: int, sessions_per_user: int, master_seed: int, output_dir: Path):
    rng = np.random.RandomState(master_seed)
    log.info(f"Generating {name} split: {n_users} users × {sessions_per_user} sessions")
    
    dataset = np.full((n_users, sessions_per_user, len(KEYBOARD_FEATURES)), np.nan, dtype=np.float32)
    
    for user_idx in range(n_users):
        style_params = _make_typing_style(rng)
        
        for sent_idx in range(sessions_per_user):
            phrase = _PHRASES[(user_idx + sent_idx) % len(_PHRASES)]
            seed = master_seed + user_idx * 1000 + sent_idx
            
            feats = _generate_session(phrase, style_params, seed)
            
            for f_idx, f_name in enumerate(KEYBOARD_FEATURES):
                dataset[user_idx, sent_idx, f_idx] = feats[f_name]
                
        if (user_idx + 1) % 500 == 0:
            log.info(f"  ... generated {user_idx + 1}/{n_users} users")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_data.npy"
    np.save(out_path, dataset)
    log.info(f"Saved {name} dataset to {out_path} with shape {dataset.shape}")
    
    with open(output_dir / "features.json", "w") as f:
        json.dump(KEYBOARD_FEATURES, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    output_dir = _BACKEND / "evaluation" / "ml" / "data"
    
    generate_split("train", 3000, 10, master_seed=100000, output_dir=output_dir)
    generate_split("val", 1000, 10, master_seed=200000, output_dir=output_dir)
    generate_split("test", 1000, 10, master_seed=300000, output_dir=output_dir)
    
    log.info("ML Dataset generation complete.")
