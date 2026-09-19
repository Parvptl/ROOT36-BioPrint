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
| Does this person behave like the enrolled user? | Behavioural fingerprint |
| Is this a person at all? | Automation detector |
| Did this capture answer *this* challenge, just now? | Integrity / replay checks |

The outcome is `ALLOW` or `BLOCK`. There is no OTP, no email, no SMS and no
second channel anywhere in the codebase.

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
│   ┌──────────┬────────────┬───────────┐                                       │
│   │ Identity │ Automation │ Integrity │                                       │
│   └──────────┴────────────┴───────────┘                                       │
│         ▼                                                                     │
│     risk engine  ──▶  ALLOW / BLOCK  + reason codes                           │
└───────────────────────────────────────────────────────────────────────────────┘
     │                                    │
     ▼ categories only                    ▼ exact scores
  login page                         operator console (key-gated)
```

The browser never computes a score. Patching the client JavaScript does not move
the authentication outcome, because the server recomputes everything from the
raw event stream.

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

### Threshold calibration

Derived, not picked. Leave-one-session-out over enrollment rounds gives genuine
scores with no session ever scored against a baseline containing it; population
samples give impostor scores. With both, the cut point is swept. With only
genuine data the threshold comes from genuine spread and the profile **records
that false acceptance is unmeasured**. With neither it reports `fallback_prior`
rather than implying a calibration happened.

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
    where threshold′ = threshold × (1 − 0.5 × automation)
```

Integrity and automation are categorical — a reused nonce is not 40% of a
replay. Below the automation gate the two signals genuinely fuse: a somewhat
machine-like attempt is held to a proportionally tighter identity threshold, so
no single number decides a login.

## 10. Security and privacy

**The login response is not an oracle.** It returns the decision, a headline, an
integrity PASS/FAIL, per-category signal bands and latency — but **no identity
score, no automation score and no threshold**. An earlier version returned all
three, which let an attacker holding a correct password read their exact distance
from acceptance and hill-climb. The exact numbers go to the audit trail and the
key-gated operator console instead.

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
| Credential (Argon2id) | 49.70 ms | 57.70 ms |
| Validation + integrity | 0.47 ms | 0.65 ms |
| Feature extraction | 0.72 ms | 0.89 ms |
| Identity scoring | 0.04 ms | 0.06 ms |
| Automation detection | 0.85 ms | 1.00 ms |
| Persistence | 0.07 ms | 0.13 ms |
| **Behavioural analysis** | **2.09 ms** | **2.42 ms** |
| **End to end** | **52.43 ms** | **60.08 ms** |

Argon2id is **95%** of the total, and is a deliberate cost. Network time is not
included and is not claimed.

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
at `/enroll`, the operator console at `/security`.

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

**Operator console** at `/security` — set `BIOPRINT_OPERATOR_KEY` in
`backend/.env`, restart, and enter it. Disabled and returns 404 without one.

## 14. Evaluation

```bash
cd backend

# Synthetic mechanism validation
./.venv/Scripts/python.exe ../evaluation/reliability_sweep.py
./.venv/Scripts/python.exe ../evaluation/diagnose_content_effect.py
./.venv/Scripts/python.exe ../evaluation/benchmark_latency.py

# Labelled evaluation against a running server — the only real-accuracy path
./.venv/Scripts/python.exe ../evaluation/run_evaluation.py \
    --username <enrolled-user> --password <their-password>
```

`run_evaluation.py` walks four phases. It prompts for the two human ones, drives
the bot and replay phases itself, and **refuses to compute a rate from fewer than
five observations**.

To collect real data:

1. **Enroll** yourself through the UI (8 rounds).
2. **Genuine** — log in ~10 times, behaving normally.
3. **Impostor** — a *different person* logs in with your correct password, ~10 times.
4. **Bot / replay** — the harness runs these automatically.

## 15. Testing

```bash
cd backend && ./.venv/Scripts/python.exe -m pytest
```

**140 tests.** Ordering is randomised (`pytest-randomly`) and warnings are errors.
Covers feature semantics, profile fitting, scale floors, saturation, calibration
paths, automation detection including false-positive cases, all four demo
scenarios, replay and nonce reuse, the oracle-removal guarantees, and assertions
that no raw events or plaintext passwords reach the database.

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
- **No adaptive profiles yet** — behaviour drifts over time and the profile does
  not follow it.
- Small samples throughout. Every rate is reported with its *n*.

## 17. Future work

Adaptive profile updates on high-confidence accepts; a larger population prior;
per-modality thresholds; a Chrome MV3 packaging of the same collector; keystroke
features conditioned on digraph identity once enough data exists to estimate
them.

## 18. Project structure

```
backend/
  app/
    api/           routes (auth, enrollment, login, ops), pipeline, rate limiting
    auth/          Argon2 passwords, sessions, challenges
    behavioral/
      features/    registry + keyboard / pointer / interaction extractors
      fingerprint/ population prior, profile fitting, scoring, calibration
      bot_detection/
      scoring/     reason codes, integrity, risk engine
    db/            schema, repository
    models/        event contract, API schemas
  tests/           140 tests
frontend/
  src/
    collector/     the behavioural sensor (TypeScript)
    pages/         login, enroll, security console
evaluation/
  reliability_sweep.py       synthetic FRR/FAR across many enrollments
  diagnose_content_effect.py content-vs-behaviour variance decomposition
  benchmark_latency.py       staged latency measurement
  run_evaluation.py          labelled real evaluation
  captures/                  real CDP automation capture
docs/
```

## 19. Authorship

Built during the ROOT 36 event window (12:00 Sat 19 Sep – 23:59 Sun 20 Sep 2026).
All behavioural logic, feature design, scoring, calibration, detection and
integration are the team's own work, authored in-window. Dependencies are
standard public packages listed in `requirements.txt` and `package.json`.
Development used an AI coding assistant, which the rules permit.
