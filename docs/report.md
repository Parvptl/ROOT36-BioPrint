# BioPrint: Behaviour-Based Login Security

**ROOT 36 / IAC 8.0, IIT Palakkad — Event 2**

### 1. Problem

Once an attacker holds a valid password, a conventional login has nothing left to
check. The standard remedy is a second channel — an OTP or email link — which
this challenge rules out. BioPrint adds a factor that cannot be stolen alongside
the password: how the legitimate user types and moves.

### 2. Threat model

**A1** password holder with different behaviour · **A2** scripted automation ·
**A3** replay of a recorded session · **A4** forged behavioural payload ·
**A5** enrollment poisoning.

We claim raised attacker cost, not impossibility. This is a probabilistic signal
and is treated as one throughout.

### 3. Solution

The browser is a **sensor**: it ships raw events and nothing else — no score, no
feature vector, no decision. The FastAPI backend is the **trust boundary** and
recomputes everything, so patching the client cannot move the outcome.

Four judgements fuse into one verdict: credential (Argon2id), behavioural
identity, automation likelihood, capture integrity. The result is `ALLOW` or
`BLOCK`. There is no OTP, email, SMS or second channel anywhere in the codebase.

### 4. Behavioural fingerprint

**27 features** across keyboard, pointer and form interaction, each declared once
in a registry with its calculation, minimum observation count and noise floor.
Distinctive picks: **negative-flight fraction** (does the next key go down before
the previous comes up — fluent typists overlap constantly, hunt-and-peck typists
never do), **shift-hand preference**, and **Tab-versus-click navigation**.

Identity keystroke features are computed over the **randomised challenge phrase
only**, so they describe how a person types rather than what they typed, and
survive a password change. The password field feeds automation detection alone.

The model is a **shrinkage-regularised robust deviation score**:

```
scale_f = max(1.4826·MAD_f·inflate(n), α·σ_pop,f, β·|median_f|, noise_floor_f)
w_f     = σ²_pop,f / (σ²_pop,f + scale²_f)
score   = Σ w_f·min(z_f², 9) / Σ w_f / 9
```

Three elements carry it: a **shrinkage floor** (a user consistent on one feature
otherwise gets a near-zero scale and is locked out), a **discriminability weight**
(uniform absent population data, rather than inventing a ranking), and
**saturation** (one odd axis cannot outvote the rest).

The threshold is **derived, not chosen**: leave-one-session-out over enrollment
gives genuine scores, population samples give impostor scores, and the cut is
swept. Where data is insufficient the profile records `genuine_only` or
`fallback_prior` and states that false acceptance is unmeasured.

### 5. Automation detection

A separate component with its own score and reason codes — *"not you"* and *"not
a person"* are different findings. Seven signals combine by **noisy-OR**:
event-order violations (an `input` with no keydown is `element.value = "…"`),
dwell degeneracy, variability collapse, timing quantisation, pointer geometry,
impossible speed, and self-declared markers.

`navigator.webdriver` and `isTrusted` are weighted 0.35 and **cannot block on
their own**; tests assert this. Vindicated by a real capture: CDP-driven
automation attacking the live page carried `isTrusted: true` and
`navigator.webdriver: false`, and was caught on event ordering at **0.95**. Kept
as a regression test.

False positives matter equally: no pointer activity is explicitly not a signal,
and pasted fields are exempt from the event-order check.

### 6. Replay protection

Each attempt binds to a single-use, expiring nonce carrying a freshly generated
phrase. The nonce is burned **before** validation, so a rejected attempt still
costs it — otherwise the endpoint becomes a tuning oracle. A recording answers
the wrong prompt; a capture describing 90 s of typing cannot belong to a
4-second-old challenge.

### 7. Authentication decision

Hard gates (integrity → automation → coverage), then a fused decision in which a
sub-blocking automation score proportionally tightens the identity threshold.
Insufficient coverage yields `BLOCK` with a retry prompt — asking for more of the
*same* behaviour, not a second factor.

**The response is not an oracle.** It returns the decision, a headline, integrity
PASS/FAIL and per-category bands — but no score and no threshold. An earlier
version returned all three, letting an attacker with the password read their
exact distance from acceptance and hill-climb. Exact values go to the audit trail
and a key-gated operator console.

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

Fitted on identical data, the obvious ML alternatives all lose — and the
comparison favours them, since missing features are filled with the enrollment
median rather than treated as uncovered:

| BioPrint | EllipticEnvelope | LOF | IsolationForest | OneClassSVM |
|---|---|---|---|---|
| **7.1%** EER | 14.8% | 17.3% | 17.5% | 17.9% |

Scripted phases against a seeded account: **bot block 100% (n=6)**, all reported
as automation; **replay rejection 100% (n=8)**, all via integrity checks.

This methodology found a real defect. `kbd_backspace_rate` moves in steps of
0.02 — one backspace in a 50-key phrase. Five rounds gave a MAD of 0.0008,
flooring the scale at 0.004, so one extra backspace scored **nine sigma out**;
that single feature carried 23% of a genuine user's deviation. The per-feature
noise floors exist because of this measurement.

**Real-human accuracy is PENDING and unmeasured.** It requires people at a
keyboard. No figure here is a real-human rate, and none has been invented.

### 10. Latency — measured, 60 decisions, warm-up discarded

| | p50 | p95 |
|---|---|---|
| **Behavioural analysis** | **2.09 ms** | **2.42 ms** |
| Credential (Argon2id) | 49.70 ms | 57.70 ms |
| **End to end** | **52.43 ms** | **60.08 ms** |

Argon2id is 95% of the total and is a deliberate memory-hard cost. An earlier
162 ms p95 was traced to warm-up and is not a steady-state figure.

### 11. Security and privacy

Argon2id (64 MiB); session tokens minted **only** after the behavioural check and
stored as SHA-256; parameterised SQL; per-client rate limiting; enrollment
credentials re-verified each round against poisoning; no hardcoded secrets.

The password field emits **timing and a coarse class only** — every printable
character collapses to `char`, and Shift collapses too, since `shift_left` at
position 4 would leak that character 4 was capitalised. The schema *itself*
rejects a key code in password context. Raw events live inside one function call
and are discarded; the database has no raw-event column, and tests assert neither
raw events nor passwords reach disk. No compliance claim is made.

### 12. Limitations

Real accuracy unmeasured; distributions overlap; browser-side bot detection is an
arms race against an adversary controlling the client; enrollment is
trust-on-first-use; pointer features are modality-dependent; rate limiting is
in-process; no adaptive profiles. Every rate carries its *n*.

### 13. Future work

Adaptive profile updates on high-confidence accepts; a larger population prior;
per-modality thresholds; Chrome MV3 packaging of the collector;
digraph-conditioned keystroke features once data allows.

---

*143 automated tests, randomised ordering, warnings as errors. All behavioural
logic authored within the event window.*
