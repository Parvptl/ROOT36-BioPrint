"""The scoring pipeline shared by enrollment and login.

One place where raw events become a decision, so both flows measure the same
things in the same way. Every stage is timed with perf_counter, because
"latency" is a claim that needs a number behind it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.auth.challenge import Challenge
from app.behavioral.bot_detection.detector import AutomationResult, detect_automation
from app.behavioral.events import SessionView, build_session_view
from app.behavioral.features.extractor import ExtractedFeatures, extract_features
from app.behavioral.fingerprint.profile import BehaviorProfile
from app.behavioral.fingerprint.scoring import IdentityResult, score_identity
from app.behavioral.ml.anomaly_model import AnomalyScore, BehavioralAnomalyModel
from app.behavioral.ml.hybrid import HybridIdentity, combine
from app.behavioral.ml.windows import window_feature_dicts
from app.behavioral.scoring.integrity import IntegrityResult, check_integrity
from app.models.events import BehaviorSessionIn
from app.models.schemas import LatencyBreakdown

log = logging.getLogger("bioprint.pipeline")


@dataclass
class PipelineResult:
    view: SessionView
    features: ExtractedFeatures
    integrity: IntegrityResult
    automation: AutomationResult
    identity: IdentityResult
    latency: LatencyBreakdown
    # Present whenever a model was supplied; None when the account has none.
    anomaly: AnomalyScore | None = None
    # The blended identity the risk engine acts on. Falls back to the
    # statistical score alone when the ML layer does not apply.
    hybrid: HybridIdentity | None = None


def run_pipeline(
    session: BehaviorSessionIn,
    challenge: Challenge,
    profile: BehaviorProfile | None,
    model: BehavioralAnomalyModel | None = None,
) -> PipelineResult:
    """Reshape, validate, extract and score one capture.

    The raw event stream exists only inside this call. Once features are out,
    nothing downstream can see it, and nothing writes it anywhere.
    """
    started = time.perf_counter()

    view = build_session_view(session)
    integrity = check_integrity(challenge, view, session.phrase_typed)
    after_validation = time.perf_counter()

    features = extract_features(view)
    after_extraction = time.perf_counter()

    identity = (
        score_identity(profile, features.values)
        if profile is not None
        else IdentityResult(score=0.0)
    )
    after_identity = time.perf_counter()

    automation = detect_automation(view, session)
    after_automation = time.perf_counter()

    # Inference only. The model is fitted at enrollment and never updated from
    # a login attempt, so nothing an attacker submits can move the baseline.
    anomaly: AnomalyScore | None = None
    if model is not None:
        try:
            anomaly = model.anomaly_for(window_feature_dicts(session))
        except Exception:  # noqa: BLE001 - a model fault must not deny service
            log.warning("anomaly scoring failed; using statistics alone", exc_info=True)
            anomaly = None
    after_ml = time.perf_counter()

    hybrid = combine(identity.score, anomaly)

    return PipelineResult(
        view=view,
        features=features,
        integrity=integrity,
        automation=automation,
        identity=identity,
        anomaly=anomaly,
        hybrid=hybrid,
        latency=LatencyBreakdown(
            total_ms=(after_ml - started) * 1000.0,
            validation_ms=(after_validation - started) * 1000.0,
            extraction_ms=(after_extraction - after_validation) * 1000.0,
            identity_ms=(after_identity - after_extraction) * 1000.0,
            automation_ms=(after_automation - after_identity) * 1000.0,
            ml_inference_ms=(after_ml - after_automation) * 1000.0,
        ),
    )
