# BioPrint — Behaviour-Based Login Security

**ROOT 36 / IAC 8.0, IIT Palakkad — Event 2 (BioPrint)**

A login system that authenticates on *how* a person interacts with the page, not
only on the credential they supply. A correct password is necessary but not
sufficient.

---

## 1. Problem

Passwords leak. Once an attacker has one, a conventional login has nothing left
to say — the credential matched, so the door opens. The usual answer is a second
channel (an OTP, an email link), which this challenge explicitly rules out.

BioPrint adds a second factor that the attacker cannot steal along with the
password: the way the legitimate user types and moves.

## 2. Solution

Three independent judgements, fused into one verdict:

| Question | Component |
|---|---|
| Does this credential match? | Argon2id password check |
| Does this person behave like the enrolled user? | Personalised statistical fingerprint |
| Is this a person at all? | Automation detector |
| Did this capture answer *this* challenge, just now? | Integrity / replay checks |

The outcome is `ALLOW` or `BLOCK`. There is no OTP, no email, no SMS and no
second channel anywhere in the codebase.

Identity is answered by one layer: a **personalised statistical fingerprint**
(medians, MADs, per-feature deviation against the enrolled user's own profile).

> **A per-user Isolation Forest was built, measured and retired.** It is not
> part of the authentication pipeline. It answered "is this behaviour unusual
> for people in general?" when the system needs "is this the enrolled user?",
> it measured worse at every blend weight, and a 60-scenario A/B showed it
> could not change a single verdict while costing ~5.6 ms of every login. See
> §6.1 and `backend/app/behavioral/ml/__init__.py`.

The **Aalto corpus is a population prior, not an identity model.** It supplies
one thing: how much humans in general vary on each keyboard feature, used to
scale a personal profile. No participant identity is ever learned, and nothing
is trained on it.

## 3. Threat model

| Attacker | Attack | Detection | Residual limitation |
|---|---|---|---|
| **A1** Has the password, is a different person | Log in normally with stolen credentials | Behavioural fingerprint. Their typing rhythm, navigation habits and pointer motion differ | Probabilistic. A person who genuinely types very like the victim will score closer. Measured overlap is real — see §11 |
| **A2** Runs a script | Automated form fill | Automation detector: event ordering, timing regularity, dwell degeneracy, pointer geometry | An adversary driving a real browser with timings sampled from real humans is not defeated |
| **A3** Records and replays a genuine session | Resubmit a captured event stream | Single-use nonce, per-attempt randomised phrase, capture-duration-vs-challenge-age check | Does not stop re-synthesising recorded *timing distributions* onto fresh prompt text in real time; that falls to A2's detector |
| **A4** Forges the behavioural payload | Submit hand-crafted events claiming to be human | Schema validation, server-side recomputation of every score, event-ordering forensics | A sufficiently faithful synthetic stream is indistinguishable in principle; we raise cost, not impossibility |
| **A5** Poisons the profile | Add their own behaviour to someone else's baseline | Credentials re-verified before every enrollment round; automated and low-coverage rounds rejected; enrollment closes once a profile exists | An attacker with the password *during* initial enrollment can enroll themselves. Enrollment is a trust-on-first-use moment |

**Explicit non-claims.** BioPrint is not unspoofable, not unhackable, and not a
perfect biometric. It raises the cost of credential-only attacks by requiring
behavioural consistency, and it detects several classes of scripted interaction.
Both are probabilistic.

## 4. Architecture

```
  Browser  =  sensor only
     │  raw events. No score, no feature vector, no decision.
     ▼
┌─────────────────────────── FastAPI: the trust boundary ───────────────────────┐
│  credential gate (Argon2id)                                                   │
│         ▼                                                                     │
│  challenge gate (nonce: single-use, expiring, bound to the claimed username)  │
│         ▼                                                                     │
│  server-side feature extraction  ──▶  raw events discarded here               │
│         ▼                                                                     │
│   ┌─────────────────────────────┬────────────┬───────────┐                    │
│   │          IDENTITY           │ Automation │ Integrity │                    │
│   │   personalised statistical  │  "is this  │ "is this  │                    │
│   │   fingerprint: median/MAD   │  a human   │  capture  │                    │
│   │   deviation from THIS       │  at all?"  │  fresh?"  │                    │
│   │   user's enrolled profile   │            │           │                    │
│   │   "is this YOU?"            │            │           │                    │
│   └─────────────────────────────┴────────────┴───────────┘                    │
│         ▼                                                                     │
│     risk engine  ──▶  ALLOW / BLOCK  + reason codes                           │
└───────────────────────────────────────────────────────────────────────────────┘
     │                                    │
     ▼ categories only                    ▼ exact scores
  login page
```

The browser never computes a score. Patching the client JavaScript does not move
the authentication outcome, because the server recomputes everything from the
raw event stream.

### The two things that must not be confused

```
  Aalto corpus                       The user
  168,593 participants               2 enrollment captures
        │                                  │
        ▼                                  ▼
  population variability            personal centre
  "how much do people differ        "what does THIS
   on each feature?"                 person do?"
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              personalised profile
                       │
                       ▼
          COLD_START / WARMING  ──►  genuine high-confidence
                       │             logins adapt the profile
                       ▼                      │
                    MATURE  ◄─────────────────┘
```

**Aalto supplies the scale, never the identity.** It is a lookup of per-feature
population spread. No participant is ever learned, no classifier is trained on
it, and it cannot recognise anyone. The centre — the part that says who you are
— comes only from the account's own captures.

The **cumulative enrollment anchor** bounds how far adaptation can move that
centre from where enrollment put it, so a run of wrongly-accepted sessions
cannot walk a profile onto an attacker (§9).

## 5. Behavioural signals

**27 features across three modalities**, each declared once in
[`registry.py`](backend/app/behavioral/features/registry.py) with its
description, calculation, minimum observation count, and noise floor.

**Keyboard (16)** — dwell median/MAD, per-hand dwell, flight median/IQR,
**negative-flight fraction** (key overlap, which separates touch typists from
hunt-and-peck), log inter-key median/MAD, burst fraction, pause rate, backspace
rate, latency-to-backspace, typing speed, speed variability, **shift-hand
preference**.

**Pointer (7)** — velocity median/p90, peak acceleration, path straightness,
direction-reversal rate, tremor, click dwell.

**Interaction (4)** — time to first interaction, **Tab-vs-click navigation
ratio**, focus-to-first-keystroke latency, pre-submit hesitation.

### Password independence

Identity keystroke features are computed over the **challenge-phrase context
only**. The phrase is regenerated every attempt, so these describe how a person
types rather than what they typed, and the profile survives a password change.
The password field is captured but feeds **automation detection only** — never
identity.

### Two rules the registry enforces

1. **A feature with too few observations is absent, never zero.** Substituting
   zero for "not observed" is the classic failure: a user who never touched the
   mouse gets pointer-velocity 0, which reads as an extreme deviation, and a
   genuine login is rejected for a reason nobody can explain. Coverage is
   tracked instead.

2. **Every feature declares a noise floor** — the smallest change that means
   different behaviour rather than the sampling resolution of one session. This
   exists because of a measured failure, documented in §11.

## 6. Fingerprint algorithm

A shrinkage-regularised robust deviation score. Per feature the profile stores a
median, a scale, and a weight.

```
z_f     = (x_f − median_f) / scale_f
scale_f = max( 1.4826·MAD_f·inflate(n),      ← the user's own spread
               α·σ_population,f,              ← how much people differ
               β·|median_f|,                  ← relative floor
               noise_floor_f )                ← measurement resolution
w_f     = σ²_pop,f / (σ²_pop,f + scale²_f)    ← discriminability
ρ(z)    = min(z², 9)                          ← saturation at 3 scale units
score   = Σ w_f·ρ(z_f) / Σ w_f / 9            ← in [0, 1]
```

**Why not One-Class SVM or Isolation Forest?** Because we fitted them on exactly
the same data and they lose. `evaluation/baseline_comparison.py`, 30 trials, same
8 enrollment rounds, same attempts, identical conditions:

| model | EER | FRR@EER | FAR@EER |
|---|---|---|---|
| **BioPrint (shrinkage robust)** | **7.1%** | 7.1% | 7.1% |
| EllipticEnvelope | 14.8% | 15.8% | 13.8% |
| LocalOutlierFactor | 17.3% | 17.9% | 16.7% |
| IsolationForest | 17.5% | 17.5% | 17.5% |
| OneClassSVM (rbf) | 17.9% | 17.9% | 17.9% |

Roughly half the error rate of the best baseline. Missing features are filled
with the enrollment median, which *favours* the baselines — they cannot express
"not observed" the way the coverage mechanism can. Eight samples in 27 dimensions
is simply not enough for these estimators; they fit enrollment noise.

Three pieces carry the model:

- **Shrinkage floor** — without it, a user who happened to be consistent on one
  feature gets a near-zero scale, every later genuine login scores enormous
  deviation on that axis, and the account is unusable.
- **Discriminability weight** — a feature earns influence when the user is tight
  on it relative to how far people spread. With no population data the weights
  go uniform, because claiming a ranking from nothing would be inventing
  information.
- **Saturation** — a single anomalous axis (borrowed mouse, sore wrist) cannot
  outvote every other feature combined.

### Enrollment: two captures

**The product asks for two natural interactions, not eight.** Eight dedicated
rounds is about two minutes of typing before anyone has logged in once, which
is real abandonment. The shortest defensible enrollment was measured rather
than guessed — `evaluation/enrollment_size_ablation.py`, 120 generated typists,
1200 genuine and 1200 impostor attempts per condition, the real Aalto prior held
fixed, thresholds calibrated on 60 users and reported on 60 disjoint held-out
users:

| captures | ROC-AUC | EER | FRR @ matched 10% FAR budget | FAR |
|---|---|---|---|---|
| 1 | 0.9758 | 8.50% | 9.67% | 9.83% |
| **2** | **0.9845** | **6.17%** | **6.50%** | **8.83%** |
| 8 *(research)* | 0.9977 | 2.08% | 0.00% | 10.00% |

Two captures beat one on **both** axes, at every budget swept (2, 5, 10, 15%) —
dominance rather than a trade-off. That is why the product asks for two.

Eight remains substantially stronger than either, which is the honest shape of
this: two captures is a usable starting point, not a mature profile. It is why
the cold-start threshold is tighter than a calibrated one, and why adaptation
graduates the profile as genuine logins arrive. The 8-round path is preserved
as the research baseline and is still reachable.

**How a two-capture profile is fitted.** The centre is the median of the two
observations; the scale still comes from the population prior. A dispersion
estimated from two points is not a dispersion — the MAD of two nearby draws
collapses toward zero and would make ordinary variation score many scale units
out. So the centre is personal from the first day and the spread is borrowed
until there is enough evidence to replace it (`profile.fit_early_profile`).

**Cold-start threshold.** 0.14, tightened from 0.20. The old value was chosen
for the single-capture distribution and measured ~28% false acceptance: a fresh
account admitting a quarter of strangers who hold the password is not an
operating point. Recalibrated against a 10% false-acceptance budget on disjoint
users across three seeds (derived cuts 0.129 / 0.143 / 0.140):

| threshold | FRR | FAR |
|---|---|---|
| 0.20 | 0.2% | 34.8% |
| **0.14** | **4.6%** | **11.9%** |

It costs roughly one retry in twenty during a state that ends when the profile
matures, and there is no account lockout. The coupling was measured too: HIGH
confidence is defined relative to the threshold, so tightening it also slows
maturation — median 7 genuine logins to graduate instead of 6, p90 of 11, and
**0 of 60 profiles failed to mature** (`evaluation/maturation_speed.py`).

Generated typists throughout. Not a claim about real-human accuracy.

### Threshold calibration

Derived, not picked. Leave-one-session-out over enrollment rounds gives genuine
scores with no session ever scored against a baseline containing it; population
samples give impostor scores. With both, the cut point is swept. With only
genuine data the threshold comes from genuine spread and the profile **records
that false acceptance is unmeasured**. With neither it reports `fallback_prior`
rather than implying a calibration happened.

When the ML blend is enabled, the threshold is **re-derived from the blended
score** by the same procedure, and `threshold_source` gains a `+ml` suffix.
Blending without recalibrating would silently move the operating point.

## 6.1 ML anomaly layer — BUILT, MEASURED, RETIRED

Full write-up: **[docs/ml-layer.md](docs/ml-layer.md)**.

### Why an anomaly detector, not a classifier

Enrollment observes exactly one class. We see how the account owner behaves and
nothing else — there is no labelled set of impostors for this user and never
will be at enrollment time. A supervised classifier has nothing to learn from.
The only question the data can answer is *how unusual is this relative to what
we have seen from this person*, which is one-class anomaly detection.

> **Retired from production.** Everything below is the record of an experiment
> that produced a negative result, kept because the result is the useful part.
> The component is not in the login path. Do not reintroduce it.
>
> Three findings retired it:
> 1. **Wrong question.** Anomaly detection asks "is this unusual?"; authentication
>    needs "is this you?" A genuine user having an odd day is anomalous but
>    authentic; an impostor sitting in the population's dense region is
>    unremarkable but wrong.
> 2. **Measured worse, monotonically** — every blend weight including the ML term
>    scored worse than excluding it (table below). It shipped at weight 0.0.
> 3. **Provably inert.** `evaluation/ml_independence.py` ran 60 scenarios —
>    enrollment, genuine, human impostor, scripted bot, value-injection bot,
>    wrong password, nonce reuse, stale capture, adaptation trail, final profile
>    state — at both 2- and 8-capture enrollment, with the model enabled and
>    bypassed. **Zero differences.** A component that cannot affect the outcome
>    is not a security layer; it is latency (~5.6 ms p50, 72% of behavioural
>    analysis time).

**What Isolation Forest does not do:** it does not identify an attacker. It
scores how easily a point is isolated from a training distribution. *Unusual*
is not *malicious* — a genuine user on a new keyboard is also unusual. That is
why it is an input to a risk engine, never a verdict.

### Windowing — making a training set out of 8 logins

One login produces one feature vector. Eight rounds in 27 dimensions is not a
training set; fitting on that fits the noise.

Each capture is cut into **overlapping windows of 20 consecutive phrase
keystrokes, stride 10**, and each window goes through *the existing feature
extractor* — there is no second feature implementation.

| | |
|---|---|
| Windows per round | ~4 |
| Training vectors per user | **~32** |
| Features at window scale | **13** |

Cut over **keystrokes, not time**: a fixed time window holds fifty presses
during fast typing and four during a pause, making vectors non-comparable. Cut
over the **phrase field only**, so no password material reaches the model. Two
features are excluded when windowed because their meaning is session-level
(`int_time_to_first_interaction` becomes the window's own offset, teaching the
model our windowing rather than the user).

Enrollment discards raw events by design, so windows are computed **at
submission time** while events are still in memory and stored as derived
features. The privacy property is unchanged.

### Model, scoring and persistence

`IsolationForest`, 150 trees, `contamination="auto"`, `random_state=20260919`
(fixed, so identical data gives an identical model), `n_jobs=1`.

Preprocessing uses a **fixed, versioned schema** decided at training and
replayed exactly at inference — the failure this prevents is silent, not loud.
Missing features are imputed to the **training centre, not zero**; zero is an
extreme value for most of these features and would manufacture an anomaly out
of a missing measurement.

`score_samples` returns higher-is-more-normal in a meaningless narrow band, so
both the inversion and the scale are fixed together:

```
anomaly = clamp( (train_centre − raw) / (3 × train_scale), 0, 1 )
```

Distance *below* the training centre in robust scale units of the training
distribution. 0 = typical, 1 = three robust sigmas out. The saturation point
matches the statistical layer's, so the two scores share a scale and a blend is
meaningful. Per-window scores aggregate by **median**.

Models persist to `data/models/<user_id>/` as `behavioral_model.joblib` plus a
readable `metadata.json` (model type, versions, feature names, training window
count, timestamp, configuration — **no user id, username or secrets**). A
corrupt, missing or version-mismatched model **degrades to the statistical
layer** rather than raising into the authentication path.

### Measured result: it does not help

`evaluation/hybrid_comparison.py` — 25 enrollments, 200 genuine + 200 impostor
attempts, identical data, equal-error (threshold-free):

| configuration | EER |
|---|---|
| **statistical only (1.0 / 0.0)** | **7.8%** |
| hybrid (0.8 / 0.2) | 9.5% |
| hybrid (0.6 / 0.4) | 12.0% |
| ML only (0.0 / 1.0) | 15.5% |

**Every weight including ML is worse, monotonically.** Alternative fusions do
not rescue it: `max` 16.2%, `min` 6.9%, statistical alone 6.2% on that run.

**Why it loses:**

| | statistical | ML |
|---|---|---|
| raw gap (impostor − genuine median) | 0.355 | **0.679** |
| genuine spread (IQR) | **0.062** | 0.298 |

A *larger* raw separation but **5× the spread on genuine users** — noisy on
precisely the class we have most data about. Worse, it correlates with the
statistical score (+0.37 genuine, +0.79 impostor) and **fails on the same
cases**: genuine users the statistical layer falsely rejects score 0.395 on ML
(also "anomalous"); impostors it falsely accepts score 0.247 (also "fine"). It
compounds errors rather than catching them. An independent signal would look
uncorrelated and be right where the other is wrong.

Worth recording: **the windowing was the right idea.** An earlier benchmark fit
Isolation Forest on 8 session-level vectors at 17.5% EER; windowing to ~32
improves it to 15.5%. It just does not close a gap that large.

### What ships

`BIOPRINT_ML_WEIGHT` defaults to **0.0**. Shipping 0.6/0.4 would knowingly ship
a **54% relative increase in equal error**. To blend it in anyway:

```bash
BIOPRINT_STATISTICAL_WEIGHT=0.6
BIOPRINT_ML_WEIGHT=0.4
```

To disable the layer entirely: `BIOPRINT_ML_ENABLED=false`.

The model is fitted **only at enrollment**, never from a login attempt, so
nothing an attacker submits can move the baseline. The ML score is withheld
from the login response for the same reason the identity score is — it would be
a tuning oracle — and appears only in the audit trail.

## 7. Automation detection

A separate component with its own score and reason codes, because *"this is not
you"* and *"this is not a person"* are different findings.

Deliberately **not** built on `navigator.webdriver` or `isTrusted`. Both are
included at weight 0.35 as corroboration only, and there are tests asserting
neither can block on its own.

Seven signals, combined with a **noisy-OR** so independent tells compound rather
than averaging out: event-order violations (an `input` with no keydown before it
is `element.value = "..."`), dwell degeneracy, timing-variability collapse,
timing quantisation, pointer geometry, physiologically impossible speed, and
self-declared markers.

**False positives are treated as equally important.** No pointer activity is
explicitly *not* a signal — filling a form from the keyboard is normal. Pasted
fields are excluded from the event-order check, because password managers exist.

### Real-world evidence

[`evaluation/captures/cdp_automation_2026-09-19.json`](evaluation/captures/cdp_automation_2026-09-19.json)
is a genuine capture from a CDP-driven browser attacking the live enrollment
page. Every event has `isTrusted: true` and `navigator.webdriver: false` — a
detector gated on either would have let it through. It was caught on event
ordering and scored **0.95**. Kept as a permanent regression test.

## 8. Replay defence

Every attempt is bound to a challenge that is single-use, expiring, and carries a
freshly generated phrase.

- **Nonce reuse** — burned atomically on first presentation, *before* validation,
  so a rejected attempt still costs the nonce. Otherwise the endpoint becomes a
  tuning oracle for refining mimicry.
- **Stale capture** — a recording answers the wrong prompt, because the phrase is
  regenerated each attempt.
- **Spliced capture** — a stream describing 90 seconds of typing cannot have been
  produced against a challenge issued 4 seconds ago.

Phrase matching is fuzzy (80% similarity): a genuine user makes typos, and a
replayed seven-word phrase scores far below that anyway.

## 9. Risk engine

Hard gates, then a fused decision.

```
integrity failed        → BLOCK (replay / integrity)
automation ≥ 0.50       → BLOCK (automation)         ← its own verdict
coverage < 0.45         → BLOCK (insufficient signal) ← declines to guess
identity > threshold′   → BLOCK (behavioural mismatch)
otherwise               → ALLOW

    identity   = w_stat × statistical + w_ml × ml_anomaly   (w_ml = 0.0)
    threshold′ = threshold × (1 − 0.5 × automation)
```

Integrity and automation are categorical — a reused nonce is not 40% of a
replay. Below the automation gate the two signals genuinely fuse: a somewhat
machine-like attempt is held to a proportionally tighter identity threshold, so
no single number decides a login.

The four scores stay **separate concepts** and are reported independently:

| | question |
|---|---|
| `statistical_identity_score` | does this match the enrolled profile? |
| `ml_anomaly_score` | is this unusual for this user? |
| `automation_score` | is this a person? |
| `integrity_score` | can this evidence be trusted? |

ML is never folded into automation or integrity. At the shipped weight the
identity score equals the statistical score exactly, so the decision path is
identical to the system before the ML layer existed.

## 10. Security and privacy

**The login response is not an oracle.** It returns the decision, a headline, an
integrity PASS/FAIL, per-category signal bands and latency — but **no identity
score, no automation score and no threshold**. An earlier version returned all
three, which let an attacker holding a correct password read their exact distance
from acceptance and hill-climb. The exact numbers go to the audit trail and the

| Control | Implementation |
|---|---|
| Password storage | Argon2id, 64 MiB, t=2. Never plaintext |
| Username enumeration | Unknown users burn equivalent CPU; challenge and status endpoints answer identically |
| Session tokens | 256-bit random, only SHA-256 stored. Minted **only** after the behavioural check |
| Injection | Every query parameterised; no string interpolation of user input into SQL |
| Rate limiting | Per-client sliding window on login, challenge, register, enrollment |
| Profile poisoning | Credentials re-verified per enrollment round; automated rounds rejected |
| Secrets | No hardcoded defaults; an unset secret generates an ephemeral one and warns |

**Privacy.** The password field emits **timing and a coarse class only** — every
printable character collapses to `char`, so not even character *class* survives.
Shift also collapses there, because `shift_left` at position 4 would reveal that
character 4 was capitalised; shift-hand preference is sourced from the public
challenge phrase instead. The schema itself rejects a payload carrying a key code
in password context, so a modified client cannot exfiltrate password characters
through this endpoint.

Raw events exist only inside one function call. They are extracted and dropped;
the database has no raw-event column, and there are tests asserting that neither
raw events nor plaintext passwords reach disk. Enrollment rounds are deleted once
a profile is fitted from them.

We make **no claim of regulatory compliance** — that has not been assessed.

## 11. Measured results

### Synthetic mechanism validation — NOT authentication accuracy

Generated typists with parameters we chose. These describe how the algorithm
behaves on controlled input; they say nothing about real people.
40 independent enrollments, 320 genuine + 320 impostor attempts:

| | threshold (median) | FRR | FAR |
|---|---|---|---|
| Cold start, no population prior | 0.183 | 6.2% | 24.4% |
| With population prior | 0.173 | 14.1% | 7.5% |
| Best single global cut | ~0.18 | ~8% | ~10% |

**The genuine and impostor distributions overlap.** There is no threshold giving
zero of both. Reported honestly because it is the actual behaviour of the system.

Adding the ML layer was measured, not assumed. Same data, same attempts,
equal-error:

| configuration | EER |
|---|---|
| **statistical only** | **7.8%** |
| hybrid (0.6 / 0.4) | 12.0% |
| ML only | 15.5% |

Every weight including ML is worse. The layer is retired, not shipped; see
§6.1 and [docs/ml-layer.md](docs/ml-layer.md) for the diagnosis.

Enrollment round count, chosen by measurement rather than feel:

| rounds | FRR | equal-error |
|---|---|---|
| 5 | 16.2% | 10.2% |
| **8 (chosen)** | **7.1%** | **7.7%** |
| 12 | 8.3% | 7.3% |

### A bug this found

`kbd_backspace_rate` over a 50-key phrase moves in steps of 0.02 — one
backspace. Five enrollment rounds gave 0.021, 0.040, 0.040, 0.021, 0.039, whose
MAD is 0.0008, flooring the scale at 0.004. A genuine login with one extra
backspace then scored **nine sigma out** and saturated. That one feature carried
23% of a genuine user's total deviation. The per-feature noise floors exist
because of this measurement.

### Real-human evaluation — **PENDING**

Genuine and impostor rates against real people have **not been measured**. That
requires humans at a keyboard (see §14); an agent cannot produce them, because
driving the browser programmatically is correctly detected and refused. No number
in this README is a real-human accuracy figure, and none will be invented.

### Latency — measured, 60 decisions, warm-up discarded

| stage | p50 | p95 |
|---|---|---|
| Credential (Argon2id) | 55.29 ms | 74.71 ms |
| Validation + integrity | 0.49 ms | 1.02 ms |
| Feature extraction | 0.78 ms | 1.30 ms |
| Identity scoring (statistical) | 0.05 ms | 0.12 ms |
| Automation detection | 0.87 ms | 1.30 ms |
| **ML anomaly (windowing + forest)** | **6.25 ms** | **7.64 ms** |
| Persistence | 0.11 ms | 0.14 ms |
| **Behavioural analysis** | **8.53 ms** | **10.91 ms** |
| **End to end** | **64.98 ms** | **83.89 ms** |

Argon2id is **85%** of the total, and is a deliberate cost. Network time is not
included and is not claimed.

Model loading was **18 ms per login** before caching — unpickling the forest on
every request dwarfed the 4 ms of actual inference. Models are cached in memory
keyed on the artefact's mtime, so a retrain is picked up on the next login
rather than serving a stale forest. Cached load: **0.05 ms**.

## 12. Installation

Requires Python 3.11+ and Node 18+.

**Backend**

```bash
cd backend
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Create `backend/.env` (copy `.env.example`) and generate a secret:

```bash
python -c "import secrets; print('BIOPRINT_SECRET_KEY=' + secrets.token_urlsafe(48))"
```

On macOS/Linux use `.venv/bin/python` instead of `./.venv/Scripts/python.exe`
throughout.

### Quickest path — one port, one command

Build the frontend once, then run only the backend. It serves the built app
itself, so the browser and the API share an origin and **no CORS setup is
involved**.

```bash
cd frontend && npm install && npm run build
cd ../backend && ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000>. The **dummy login page is at `/login`**, enrollment
at `/enroll`.

Confirm the bundle was picked up — `GET /health` reports
`"frontend_bundled": true`.

### Development mode — two ports, hot reload

```bash
cd backend && ./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev          # separate terminal
```

Open <http://localhost:5173>. This path *does* cross origins, which is why
`BIOPRINT_CORS_ORIGINS` defaults to the Vite dev server. If you serve the built
bundle from some other port instead, add that origin there — or just use the
single-port path above, which avoids the problem entirely.

## 13. Using it

**Enroll** at `/enroll` — set a username and password, consent, then complete
**8 short rounds** (~3 minutes). Each round is the same form as the login page,
with a different phrase. Fill it the way you normally would; an artificially neat
baseline will reject the real you later.

**Log in** at `/login` — username, password, and the phrase shown. The phrase is
different every time.

## 14. Evaluation

```bash
cd backend

# Synthetic mechanism validation
./.venv/Scripts/python.exe ../evaluation/reliability_sweep.py
./.venv/Scripts/python.exe ../evaluation/diagnose_content_effect.py
./.venv/Scripts/python.exe ../evaluation/baseline_comparison.py
./.venv/Scripts/python.exe ../evaluation/hybrid_comparison.py   # statistical vs ML
./.venv/Scripts/python.exe ../evaluation/benchmark_latency.py

# Labelled evaluation against a running server — the only real-accuracy path
./.venv/Scripts/python.exe ../evaluation/run_evaluation.py \
    --username <enrolled-user> --password <their-password>
```

`run_evaluation.py` walks four phases. It prompts for the two human ones, drives
the bot and replay phases itself, and **refuses to compute a rate from fewer than
five observations**.

To collect real data:

1. **Enroll** yourself through the UI (2 captures; the 8-round research
   baseline is still available by requesting `rounds: 8`).
2. **Genuine** — log in ~10 times, behaving normally.
3. **Impostor** — a *different person* logs in with your correct password, ~10 times.
4. **Bot / replay** — the harness runs these automatically.

Three properties of the harness worth knowing before reading its numbers:

**Password failures are separated from behavioural decisions.** A wrong password
is rejected by the credential gate before any behavioural analysis runs, so the
attempt carries no identity score and no threshold comparison. Those attempts
are reported as `PASSWORD_FAILURE` with their count and their attempt ids, and
excluded from FAR and FRR. They are never silently dropped. In the first
real-human run this mattered: three of 43 attempts were mistyped passwords, and
counting them put genuine acceptance at 85.7% when the behavioural figure was
92.3%.

**It refuses to measure a stale server.** A long-lived uvicorn process serving
code older than the source tree produced an entire evaluation run against a
build with neither the ML nor the adaptive layer in it, and nothing in the
results showed it. `/health` now reports `started_at`, and the harness exits if
any file under `backend/app` is newer.

**It never overwrites a previous run.** Output goes to a timestamped file plus
`latest.json`; earlier runs stay in `evaluation/out/` and `evaluation/out/baseline/`.

The forensic analysis of the first run is in
[docs/evaluation-run1-forensics.md](docs/evaluation-run1-forensics.md), and
`evaluation/forensics.py` dumps any run's audit trail per attempt.

## 15. Testing

```bash
cd backend && ./.venv/Scripts/python.exe -m pytest
```

**176 tests.** Ordering is randomised (`pytest-randomly`) and warnings are errors.
Covers feature semantics, profile fitting, scale floors, saturation, calibration
paths, automation detection including false-positive cases, all four demo
scenarios, replay and nonce reuse, the oracle-removal guarantees, and assertions
that no raw events or plaintext passwords reach the database.

33 of those cover the ML layer: windowing, schema versioning, train/inference
mismatch, persistence round-trip, corrupt and version-mismatched artefacts,
cache invalidation on retrain, determinism, graceful fallback when data is
insufficient, and weight renormalisation.

## 16. Limitations

- **Accuracy on real people is unmeasured.** §11 is synthetic.
- **The distributions overlap.** This is a probabilistic signal.
- **Browser-side bot detection is an arms race** against an adversary who
  controls the client.
- **Trust-on-first-use enrollment** — an attacker with the password during
  initial enrollment enrolls themselves.
- **Pointer features are modality-dependent.** Enrolling on a trackpad and
  logging in with a mouse shifts them; coverage and weighting limit the damage
  but do not remove it.
- **Rate limiting is in-process**; a multi-worker deployment needs shared state.
- **The ML layer does not currently earn its place.** It is trained and scored
  but weighted 0.0, because blending it measurably worsens separation on
  synthetic data. That conclusion may not hold on real humans and should be
  re-measured once real captures exist.
- Small samples throughout. Every rate is reported with its *n*.

## 17. Future work

A larger population prior; per-modality thresholds; a Chrome MV3 packaging of the
same collector; keystroke features conditioned on digraph identity once enough
data exists to estimate them. For the ML layer: re-measure on real captures, and
try an estimator whose errors are less correlated with the statistical layer's —
the current one adds little because it fails on the same attempts.

## 18. Project structure

```
backend/
  app/
    api/           routes (auth, enrollment, login, ops), pipeline, rate limiting
    auth/          Argon2 passwords, sessions, challenges
    behavioral/
      features/    registry + keyboard / pointer / interaction extractors
      fingerprint/ population prior, profile fitting, scoring, calibration
      ml/          RETIRED Isolation Forest experiment (not in the login path)
      bot_detection/
      scoring/     reason codes, integrity, risk engine
    db/            schema, repository
    models/        event contract, API schemas
  tests/           176 tests
frontend/
  src/
    collector/     the behavioural sensor (TypeScript)
    pages/         login, enroll
evaluation/
  reliability_sweep.py       synthetic FRR/FAR across many enrollments
  diagnose_content_effect.py content-vs-behaviour variance decomposition
  baseline_comparison.py     statistical model vs sklearn one-class baselines
  hybrid_comparison.py       statistical vs ML vs hybrid, weight sweep
  benchmark_latency.py       staged latency measurement
  run_evaluation.py          labelled real evaluation
  captures/                  real CDP automation capture
docs/
  ml-layer.md                design, measurement and shadow-mode rationale
  report.md                  the required technical report
  threat-model, judge-assessment
```

## 19. Authorship

Built during the ROOT 36 event window (12:00 Sat 19 Sep – 23:59 Sun 20 Sep 2026).
All behavioural logic, feature design, scoring, calibration, detection and
integration are the team's own work, authored in-window. Dependencies are
standard public packages listed in `requirements.txt` and `package.json`.
Development used an AI coding assistant, which the rules permit.
