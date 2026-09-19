# BioPrint: Behaviour-Based Login Security

**ROOT 36 / IAC 8.0, IIT Palakkad — Event 2**

### 1. Problem

Once an attacker holds a valid password, a conventional login has nothing left to
check. The standard remedy is a second channel — an OTP or email link — which
this challenge rules out. BioPrint adds a factor that cannot be stolen alongside
the password: how the legitimate user types and moves.

### 2. Threat model

**A1** password holder with different behaviour · **A2** scripted automation ·
**A3** replay · **A4** forged payload · **A5** enrollment poisoning. We claim
raised attacker cost, not impossibility: this is a probabilistic signal and is
treated as one throughout.

### 3. Solution

The browser is a **sensor**: it ships raw events and nothing else — no score, no
feature vector, no decision. The FastAPI backend is the **trust boundary** and
recomputes everything, so patching the client cannot move the outcome.

Four judgements fuse into one verdict: credential (Argon2id), behavioural
identity, automation likelihood, capture integrity. Identity itself is answered
by two independent layers, a statistical fingerprint and a per-user anomaly
detector. The result is `ALLOW` or `BLOCK`. There is no OTP, email, SMS or
second channel anywhere in the codebase.

### 4. Behavioural fingerprint

**27 features** across keyboard, pointer and form interaction, each declared once
in a registry with its calculation, minimum observations and noise floor.
Distinctive picks: **negative-flight fraction** (does the next key go down before
the previous comes up), **shift-hand preference**, **Tab-versus-click navigation**.

Identity keystroke features use the **randomised challenge phrase only**, so they
describe how a person types rather than what they typed and survive a password
change. The password field feeds automation detection alone.

A **shrinkage-regularised robust deviation score**:

```
scale_f = max(1.4826·MAD_f·inflate(n), α·σ_pop,f, β·|median_f|, noise_floor_f)
w_f     = σ²_pop,f / (σ²_pop,f + scale²_f)
score   = Σ w_f·min(z_f², 9) / Σ w_f / 9
```

Three elements carry it: a **shrinkage floor** (a consistent user otherwise gets
a near-zero scale and is locked out), a **discriminability weight** (uniform
absent population data, rather than inventing a ranking), and **saturation** (one
odd axis cannot outvote the rest). The threshold is **derived, not chosen**, by
leave-one-session-out; where data is insufficient the profile records that and
states false acceptance is unmeasured.

### 4b. ML anomaly layer

Enrollment observes **one class**: how the owner behaves, nothing else. No
labelled impostor set exists at enrollment time, so a supervised classifier has
nothing to learn from. The answerable question is *how unusual is this for this
person* — one-class anomaly detection, here a per-user **Isolation Forest**.

It does **not** identify an attacker; it scores how easily a point is isolated
from the training distribution. Unusual is not malicious, so it informs the risk
engine rather than deciding.

Eight logins give eight vectors in 27 dimensions, which is not a training set.
Each capture is cut into **overlapping windows of 20 phrase keystrokes, stride
10**, passed through the *existing* extractor: roughly **32 training vectors of
13 features** per user. Windows follow keystrokes rather than time, and the
phrase field only, so no password material reaches the model. Scores map to
[0,1] as distance below the training centre in robust scale units, saturating at
three like the statistical layer so a blend is meaningful.

**Measured, and it does not help.** 25 enrollments, 200 genuine and 200
impostor attempts, equal-error:

| statistical only | hybrid 0.8/0.2 | hybrid 0.6/0.4 | ML only |
|---|---|---|---|
| **7.8%** | 9.5% | 12.0% | 15.5% |

Every weight including ML is worse, monotonically. The anomaly score has a
*larger* raw gap (0.679 vs 0.355) but **five times the spread on genuine users**
(IQR 0.298 vs 0.062), and correlates with the statistical score (+0.37, +0.79) —
failing on the same attempts rather than catching them.

It ships in **shadow mode**: trained, scored, logged, shown on the operator
console, **weighted 0.0**. Shipping 0.6/0.4 would knowingly add 54% relative
equal error. One environment variable enables it and re-derives the threshold
from the blended score.

### 5. Automation detection

A separate component with its own score and reason codes — *"not you"* and *"not
a person"* are different findings. Seven signals combine by **noisy-OR**:
event-order violations (an `input` with no keydown is `element.value = "…"`),
dwell degeneracy, variability collapse, timing quantisation, pointer geometry,
impossible speed, and self-declared markers.

`navigator.webdriver` and `isTrusted` are weighted 0.35 and **cannot block on
their own**; tests assert this. Vindicated by a real capture: CDP-driven
automation carried `isTrusted: true` and `navigator.webdriver: false`, and was
caught on event ordering at **0.95**. Kept as a regression test.

False positives matter equally: absent pointer activity is explicitly not a
signal, and pasted fields are exempt from the event-order check.

### 6. Replay protection

Each attempt binds to a single-use, expiring nonce carrying a freshly generated
phrase, burned **before** validation so a rejected attempt still costs it —
otherwise the endpoint is a tuning oracle. A recording answers the wrong prompt;
a capture describing 90 s of typing cannot belong to a 4-second-old challenge.

### 7. Authentication decision

Hard gates (integrity → automation → coverage), then a fused decision where a
sub-blocking automation score proportionally tightens the identity threshold.
Insufficient coverage yields `BLOCK` with a retry prompt — more of the *same*
behaviour, not a second factor.

**The response is not an oracle**: decision, headline, integrity PASS/FAIL and
per-category bands, but no score and no threshold. An earlier version returned
all three, letting an attacker with the password read their distance from
acceptance and hill-climb. Exact values go to the audit trail and the key-gated
operator console.

### 8–9. Evaluation and results

**Synthetic mechanism validation — not authentication accuracy.** Generated
typists; 40 enrollments, 320 genuine + 320 impostor attempts.

| | FRR | FAR |
|---|---|---|
| Cold start, no prior | 6.2% | 24.4% |
| With population prior | 14.1% | 7.5% |
| Best global cut | ~8% | ~10% |

**The distributions overlap**; no threshold gives zero of both. Enrollment was set
to **8 rounds** by measurement (5 → EER 10.2%; 8 → 7.7%; 12 → 7.3%).

One-class baselines fitted on identical session-level data all lose:
EllipticEnvelope 14.8%, LOF 17.3%, IsolationForest 17.5%, OneClassSVM 17.9%,
against **7.1%**.

Scripted phases against a seeded account: **bot block 100% (n=6)**, all reported
as automation; **replay rejection 100% (n=8)**, all via integrity checks.

This methodology found a real defect: `kbd_backspace_rate` moves in steps of one
backspace per 50 keys, its MAD collapsed below that resolution, and one extra
correction scored **nine sigma out** — that feature alone carrying 23% of genuine
deviation. The per-feature noise floors exist because of this measurement.

**Real-human accuracy is PENDING and unmeasured.** It requires people at a
keyboard. No figure here is a real-human rate, and none has been invented.

### 10. Latency — measured, 60 decisions, warm-up discarded

| | p50 | p95 |
|---|---|---|
| ML anomaly (windowing + forest) | 6.25 ms | 7.64 ms |
| **Behavioural analysis** | **8.53 ms** | **10.91 ms** |
| Credential (Argon2id) | 55.29 ms | 74.71 ms |
| **End to end** | **64.98 ms** | **83.89 ms** |

Argon2id is 85% of the total, a deliberate memory-hard cost. Model loading cost
18 ms per login until models were cached on artefact mtime; cached load 0.05 ms.

### 11. Security and privacy

Argon2id (64 MiB); session tokens minted **only** after the behavioural check,
stored as SHA-256; parameterised SQL; rate limiting; enrollment credentials
re-verified each round against poisoning; no hardcoded secrets. The model is
fitted only at enrollment, so nothing an attacker submits moves the baseline.

The password field emits **timing and a coarse class only** — every printable
character collapses to `char`, and Shift too, since `shift_left` at position 4
would leak that character 4 was capitalised. The schema *itself* rejects a key
code in password context. Raw events live inside one function call and are
discarded; the database has no raw-event column, and tests assert neither raw
events nor passwords reach disk. No compliance claim is made.

### 12. Limitations

Real accuracy unmeasured; distributions overlap; browser-side bot detection is an
arms race against an adversary controlling the client; enrollment is
trust-on-first-use; pointer features are modality-dependent; rate limiting is
in-process; no adaptive profiles; the ML layer does not currently earn its
place and is weighted 0.0. Every rate carries its *n*.

### 13. Future work

Adaptive profile updates on high-confidence accepts; a larger population prior;
per-modality thresholds; Chrome MV3 packaging of the collector; and for the ML
layer, an estimator whose errors are less correlated with the statistical
layer's.

---

*176 automated tests, randomised ordering, warnings as errors. All behavioural
logic authored within the event window.*
