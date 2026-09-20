# The ML anomaly layer

An Isolation Forest anomaly detector, per user, trained at enrollment and
scored at login alongside the existing statistical fingerprint.

**It ships in shadow mode: trained, scored, logged and displayed, but weighted
0.0 in the decision.** That is a measurement result, documented below, not a
failure to finish the integration.

---

## Why an anomaly detector

Enrollment observes exactly one class. We see how the account owner behaves and
nothing else; there is no labelled set of impostors for this user and there
never will be at enrollment time. A supervised classifier has nothing to learn
from. The only question the data can answer is *how unusual is this relative to
what we have seen from this person*, which is the definition of one-class
anomaly detection.

**What Isolation Forest does not do:** it does not identify an attacker. It
scores how easily a point is isolated from a training distribution. Unusual is
not the same as malicious — a genuine user on a new keyboard is also unusual.
That is why it is an input to a risk engine and never a verdict on its own.

## Windowing: the training set problem

One login produces one feature vector. Eight enrollment rounds produce eight,
in twenty-seven dimensions. Fitting anything on that fits the noise.

Each capture is therefore cut into **overlapping windows of 20 consecutive
phrase keystrokes, stride 10**, and each window goes through *the existing
feature extractor*. There is no second feature implementation.

- windows per round: ~4
- training vectors per user: **~32**
- features surviving at window size: **13** (keystroke-dominated; pointer
  features are usually absent during typing)

Windows are cut over **keystrokes, not wall-clock time**: features here are
keystroke-driven, and a fixed time window would hold fifty presses during fast
typing and four during a pause, making the vectors non-comparable.

Windows are cut over the **challenge-phrase field only**, the same restriction
the identity features use. The phrase is regenerated per attempt, so the model
learns how the person types rather than what they typed, and nothing about the
password reaches it.

Two features are excluded when windowed:

| feature | why |
|---|---|
| `int_time_to_first_interaction` | Inside a window this is the window's own offset — it encodes *where in the session the window sits*, teaching the model our windowing rather than the user |
| `int_presubmit_hesitation` | Exists only in the window containing the submit; present in one window per session and absent in the rest |

## Preprocessing

A fixed, versioned schema decided once at training and replayed exactly at
inference. The failure this prevents is silent: a model fitted on one column
order asked to score a vector assembled differently does not error, it returns
nonsense.

- schema = features present in ≥70% of training windows
- robust standardisation (median / MAD), matching the statistical layer
- **missing features imputed to the training centre, not zero** — zero is an
  extreme value for most of these features and would manufacture an anomaly out
  of a missing measurement
- schema, centres, scales and version stored with the model; a version mismatch
  refuses the model rather than reinterpreting it

## Score

`score_samples` returns higher-is-more-normal in a narrow band near −0.45,
which means nothing on its own. Both the inversion and the scale are handled
together:

```
anomaly = clamp( (train_centre − raw) / (3 × train_scale), 0, 1 )
```

Expressed as distance *below* the training centre in robust scale units of the
training distribution. A login at the training median scores 0; three robust
sigmas below scores 1. Scoring *above* the centre clamps to 0 rather than going
negative. The saturation point of 3 matches the statistical layer's, so the two
scores speak the same dialect and a weighted blend is meaningful.

Per-window scores are aggregated by **median**, so one window covering a pause
or a correction does not drag the whole login.

## Measured result

`evaluation/hybrid_comparison.py` — 25 independent enrollments, 200 genuine and
200 impostor attempts, identical data for every configuration. Equal-error rate
is threshold-free, so this measures discriminative power rather than whose
threshold happened to sit better.

| configuration | EER |
|---|---|
| **statistical only (1.0 / 0.0)** | **7.8%** |
| hybrid (0.8 / 0.2) | 9.5% |
| hybrid (0.7 / 0.3) | 10.5% |
| hybrid (0.6 / 0.4) | 12.0% |
| hybrid (0.5 / 0.5) | 12.5% |
| hybrid (0.4 / 0.6) | 13.0% |
| ML only (0.0 / 1.0) | 15.5% |

**Every weight including the ML term is worse, monotonically.** Alternative
fusions do not rescue it: `max` 16.2%, `min` 6.9%, statistical alone 6.2% on
the same run.

### Why it loses

| | statistical | ML |
|---|---|---|
| genuine median | 0.098 | 0.151 |
| impostor median | 0.453 | 0.830 |
| raw gap | 0.355 | **0.679** |
| genuine spread (IQR) | **0.062** | 0.298 |

The ML score has a *larger* raw separation but a **five times larger spread on
genuine users** — it is noisy on precisely the class we have the most data
about, so the distributions overlap more despite the bigger gap.

Worse, it is positively correlated with the statistical score (+0.37 genuine,
+0.79 impostor) and **fails on the same cases**:

- genuine users the statistical layer falsely rejects score **0.395** on ML —
  also "anomalous"
- impostors the statistical layer falsely accepts score **0.247** on ML — also
  "fine"

It compounds the statistical layer's errors instead of catching them. An
independent signal would look different: uncorrelated, and right where the
other is wrong.

### Windowing did help the ML side

An earlier benchmark (`baseline_comparison.py`) fitted Isolation Forest on
**8 session-level vectors** and measured 17.5% EER. Windowing to ~32 vectors
improves it to 15.5%. The windowing was the right idea and it worked; it simply
does not close a gap that large.

## What ships

`BIOPRINT_ML_WEIGHT` defaults to **0.0**. The model is trained, persisted,
scored on every login, written to the audit trail and shown on the operator
console — it just does not move the decision.

Shipping 0.6/0.4 would knowingly ship a **54% relative increase in equal
error**. To blend it in anyway:

```bash
BIOPRINT_STATISTICAL_WEIGHT=0.6
BIOPRINT_ML_WEIGHT=0.4
```

Enabling it re-derives the threshold from leave-one-session-out scores of the
*blended* score, and marks `threshold_source` with a `+ml` suffix. Blending
without recalibrating would silently move the operating point. The leave-one-out
holds out a **whole round**, never individual windows — windows from one round
overlap, so splitting between them would leak held-out data back into training.

To turn the layer off entirely: `BIOPRINT_ML_ENABLED=false`.

## Latency

`evaluation/benchmark_latency.py`, 60 decisions, warm-up discarded:

| stage | p50 | p95 |
|---|---|---|
| ML anomaly (windowing + forest) | 6.25 ms | 7.64 ms |
| behavioural analysis total | 8.53 ms | 10.91 ms |
| end to end | 64.98 ms | 83.89 ms |

Model loading was **18 ms per login** before caching — unpickling the forest on
every request dwarfed the 4 ms of actual inference. Models are now cached in
memory keyed on the artefact's mtime, so a retrain is picked up on the next
login rather than serving a stale forest. Cached load: **0.05 ms**.

## Security

- The model is fitted **only at enrollment**, never from a login attempt, so
  nothing an attacker submits can move the baseline.
- Nothing reachable from a browser writes to the model store.
- Model metadata carries no user id, username, password material or secrets —
  a test asserts this.
- A corrupt, missing or version-mismatched model degrades to the statistical
  layer rather than raising into the authentication path.
- The ML score is withheld from the login response for the same reason the
  identity score is: it would be a tuning oracle. It appears only in the audit
  trail.

## Limitations

- Everything above is **synthetic mechanism validation**. Real-human accuracy is
  unmeasured for both layers.
- 32 training windows is still a small training set; the windows overlap, so the
  effective independent sample count is lower than 32.
- Pointer features are mostly absent at window scale, so the model is largely a
  keystroke model.
- Conclusions may not hold on real human data, where the statistical layer's
  genuine spread may be wider and the gap correspondingly narrower. Re-run
  `hybrid_comparison.py` once real captures exist.
