"""Saving and loading per-user anomaly models.

The model is security-sensitive: it decides, in part, who gets in. Nothing
reachable from a browser writes here. Training happens on the backend during
enrollment, and this module is the only writer.

Layout:

    <model_dir>/<user_id>/behavioral_model.joblib
    <model_dir>/<user_id>/metadata.json

joblib holds the fitted estimator and the processor. The metadata is written
separately as readable JSON so a model can be inspected, audited and
version-checked without unpickling anything.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib

from app.behavioral.ml.anomaly_model import (
    MODEL_VERSION,
    BehavioralAnomalyModel,
)
from app.behavioral.ml.preprocessing import PREPROCESSING_VERSION, SchemaMismatch
from app.config import settings

log = logging.getLogger("bioprint.ml")

MODEL_FILENAME = "behavioral_model.joblib"
METADATA_FILENAME = "metadata.json"


def _user_dir(user_id: int, model_dir: Path | None = None) -> Path:
    root = model_dir or settings.model_dir
    # int() rather than string formatting: a non-numeric user id must never
    # become a path segment, which is how directory traversal gets in.
    return root / str(int(user_id))


def save_model(
    user_id: int, model: BehavioralAnomalyModel, model_dir: Path | None = None
) -> Path:
    directory = _user_dir(user_id, model_dir)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"processor": model.processor, "forest": model.forest},
        directory / MODEL_FILENAME,
        compress=3,
    )

    _CACHE.pop(_cache_key(user_id, model_dir), None)
    metadata = model.describe()
    # The user id is not written. The directory already encodes it, and the
    # metadata file is meant to be safe to read and share on its own.
    (directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory


# Loaded models, keyed by user and validated against the artefact's mtime.
#
# Unpickling a fitted forest costs about 18 ms, which was being paid on every
# single login and dwarfed the 4 ms of actual inference. Caching the object is
# the fix; keying on mtime is what keeps it correct, because a retrained model
# writes a new file and the next login picks it up rather than serving a stale
# forest indefinitely.
_CACHE: dict[tuple[str, int], tuple[float, BehavioralAnomalyModel]] = {}
_CACHE_LIMIT = 64


def _cache_key(user_id: int, model_dir: Path | None) -> tuple[str, int]:
    root = model_dir or settings.model_dir
    return (str(root), int(user_id))


def clear_model_cache() -> None:
    """Drop every cached model. Used by tests and by the demo reset."""
    _CACHE.clear()


def load_model(
    user_id: int, model_dir: Path | None = None
) -> BehavioralAnomalyModel | None:
    """Load a stored model, or None if there is not a usable one.

    Every failure path returns None rather than raising. A missing, corrupt or
    version-mismatched model must degrade to the statistical layer, never take
    an authentication request down with it.
    """
    directory = _user_dir(user_id, model_dir)
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME

    if not model_path.is_file() or not metadata_path.is_file():
        return None

    key = _cache_key(user_id, model_dir)
    try:
        stamp = model_path.stat().st_mtime_ns
    except OSError:
        return None

    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("unreadable model metadata for user %s", user_id)
        return None

    if metadata.get("model_version") != MODEL_VERSION:
        log.warning(
            "model for user %s was built by version %s, current is %s; ignoring",
            user_id,
            metadata.get("model_version"),
            MODEL_VERSION,
        )
        return None

    if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
        log.warning(
            "model for user %s uses preprocessing %s, current is %s; ignoring",
            user_id,
            metadata.get("preprocessing_version"),
            PREPROCESSING_VERSION,
        )
        return None

    try:
        payload = joblib.load(model_path)
        processor = payload["processor"]
        forest = payload["forest"]
    except Exception:  # noqa: BLE001 - a bad artefact must not break login
        log.warning("could not load model artefact for user %s", user_id, exc_info=True)
        return None

    # The estimator and the schema travel together, but check rather than trust:
    # a mismatch here is the silent-nonsense failure mode this guards against.
    expected = metadata.get("feature_names") or []
    if list(processor.feature_names) != list(expected):
        log.warning("feature schema mismatch for user %s; ignoring model", user_id)
        return None

    try:
        if processor.version != PREPROCESSING_VERSION:
            raise SchemaMismatch(processor.version)
    except SchemaMismatch:
        return None

    model = BehavioralAnomalyModel(
        processor=processor,
        forest=forest,
        train_centre=float(metadata.get("train_centre", 0.0)),
        train_scale=float(metadata.get("train_scale", 1.0)) or 1.0,
        n_training_windows=int(metadata.get("n_training_windows", 0)),
        trained_at=float(metadata.get("trained_at", 0.0)),
        model_type=str(metadata.get("model_type", "")),
        model_version=str(metadata.get("model_version", "")),
    )

    if len(_CACHE) >= _CACHE_LIMIT:
        # Crude eviction is fine here: the cache is a latency optimisation, and
        # a wrongly evicted entry costs one reload, not a wrong answer.
        _CACHE.clear()
    _CACHE[key] = (stamp, model)
    return model


def delete_model(user_id: int, model_dir: Path | None = None) -> None:
    """Remove a user's model. Used by profile reset and the demo reset."""
    _CACHE.pop(_cache_key(user_id, model_dir), None)
    directory = _user_dir(user_id, model_dir)
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)


def delete_all_models(model_dir: Path | None = None) -> int:
    """Wipe every stored model. Returns how many were removed."""
    clear_model_cache()
    root = model_dir or settings.model_dir
    if not root.is_dir():
        return 0
    removed = 0
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def has_model(user_id: int, model_dir: Path | None = None) -> bool:
    return (_user_dir(user_id, model_dir) / MODEL_FILENAME).is_file()
