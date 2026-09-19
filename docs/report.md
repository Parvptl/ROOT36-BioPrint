# BioPrint: Behaviour-Based Login Security

**ROOT 36 / IAC 8.0, IIT Palakkad — Event 2**

---

### 1. Problem

Once an attacker holds a valid password, a conventional login has nothing left to
check. The standard remedy is a second channel — an OTP or an email link — which
this challenge rules out. BioPrint adds a factor that cannot be stolen alongside
the password: how the legitimate user types and moves.

### 2. Threat model

**A1** password holder with different behaviour · **A2** scripted automation ·
**A3** replay of a recorded session · **A4** forged behavioural payload ·
**A5** enrollment poisoning.

We claim raised attacker cost, not impossibility. BioPrint is not unspoofable and
is not a perfect biometric; it is a probabilistic signal, and it is treated as one
throughout.

### 3. Solution

The browser is a **sensor**. It ships raw events and nothing else — no score, no
feature vector, no decision. The FastAPI backend is the **trust boundary** and
recomputes everything, so patching the client JavaScript cannot move the outcome.

Four judgements fuse into one verdict: credential (Argon2id), behavioural
identity, automation likelihood, and capture integrity. The result is `ALLOW` or
`BLOCK`. There is no OTP, email, SMS or second channel anywhere in the codebase.

### 4. Behavioural fingerprint

**27 features** over keyboard, pointer and form-interaction, each declared once in
a registry with its calculation, minimum observation count and noise floor.
Distinctive choices include **negative-flight fraction** (whether the next key
goes down before the previous comes up, which separates touch typists from
hunt-and-peck), **shift-hand preference**, and **Tab-versus-click navigation**.

Identity keystroke features are computed over the **randomised challenge phrase
only**, so they describe how a person types rather than what they typed and
survive a password change. The password field feeds automation detection alone.

The model is a **shrinkage-regularised robust deviation score**: per feature a
median, a scale, and a discriminability weight; per attempt a weighted, saturated
distance in scale units.

```
scale_f = max(1.4826·MAD_f·inflate(n), α·σ_pop,f, β·|median_f|, noise_floor_f)
w_f     = σ²_pop,f / (σ²_pop,f + scale²_f)
score   = Σ w_f·min(z_f², 9) / Σ w_f / 9
```

One-Class SVM, Isolation Forest and autoencoders were rejected as unestimable at
8 samples in 27 dimensions; they would fit enrollment noise. Three elements carry
the model: a **shrinkage floor** (a user consistent on one feature otherwise gets
a near-zero scale and is locked out), a **discriminability weight** (uniform when
no population data exists, because inventing a ranking from nothing is dishonest),
and **saturation** (one odd axis cannot outvote the rest).

The threshold is **derived, not chosen**: leave-one-session-out over enrollment
gives genuine scores, population samples give impostor scores, and the cut is
swept. Where data is insufficient the profile records `genuine_only` or
`fallback_prior` and states that false acceptance is unmeasured.

### 5. Automation detection

A separate component with its own score and reason codes: *"not you"* and *"not a
person"* are different findings. Seven signals combine by **noisy-OR** so
independent tells compound — event-order violations (an `input` with no keydown
is `element.value = "…"`), dwell degeneracy, variability collapse, timing
quantisation, pointer geometry, impossible speed, and self-declared markers.

`navigator.webdriver` and `isTrusted` are weighted 0.35 and **cannot block on
their own**; tests assert this. That decision is vindicated by a real capture:
CDP-driven automation attacking the live page carried `isTrusted: true` and
`navigator.webdriver: false`, and was caught on event ordering at score **0.95**.
It is kept as a regression test.

False positives are treated as equally serious. Absence of pointer activity is
explicitly not a signal; pasted fields are exempt from the event-order check.

### 6. Replay protection

Each attempt binds to a single-use, expiring nonce carrying a freshly generated
phrase. The nonce is burned **before** validation, so a rejected attempt still
costs it — otherwise the endpoint becomes a tuning oracle. A recording answers the
wrong prompt; a spliced capture describing 90 s of typing cannot belong to a
4-second-old challenge.

### 7. Authentication decision

Hard gates (integrity → automation → coverage), then a fused identity decision in
which a sub-blocking automation score proportionally tightens the identity
threshold. Insufficient coverage yields `BLOCK` with a retry prompt — asking for
more of the *same* behaviour, not a second factor.

**The response is not an oracle.** It returns the decision, a headline, integrity
PASS/FAIL and per-category bands — but no score and no threshold. An earlier
version returned all three; an attacker with the password could read their exact
distance from acceptance and hill-climb. Exact values go to the audit trail and a
key-gated operator console.

### 8–9. Evaluation and results

**Synthetic mechanism validation — not authentication accuracy.** Generated
typists, 40 independent enrollments, 320 genuine + 320 impostor attempts:

| | FRR | FAR |
|---|---|---|
| Cold start, no prior | 6.2% | 24.4% |
| With population prior | 14.1% | 7.5% |
| Best global cut | ~8% | ~10% |

**The distributions overlap**; no threshold gives zero of both. Enrollment was set
to **8 rounds** by measurement (5 → FRR 16.2%, EER 10.2%; 8 → 7.1%, 7.7%; 12 →
8.3%, 7.3%).

Scripted phases against a seeded account: **bot block rate 100% (n=6)**, all
reported as automation; **replay rejection 100% (n=8)**, all via integrity checks.

This methodology found a real defect. `kbd_backspace_rate` moves in steps of 0.02
— one backspace in a 50-key phrase. Five rounds gave a MAD of 0.0008, flooring
the scale at 0.004, so one extra backspace scored **nine sigma out**; that single
feature carried 23% of a genuine user's deviation. Per-feature noise floors exist
because of this measurement, not because they seemed prudent.

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

Argon2id (64 MiB); session tokens minted **only** after the behavioural check;
SHA-256-hashed token storage; parameterised SQL throughout; per-client rate
limiting; enrollment credential re-verification against profile poisoning; no
hardcoded secrets.

The password field emits **timing and a coarse class only** — every printable
character collapses to `char`, and Shift collapses too, since `shift_left` at
position 4 would leak that character 4 was capitalised. The schema *itself*
rejects a key code in password context, so a modified client cannot exfiltrate
password characters. Raw events live inside one function call and are discarded;
the database has no raw-event column, and tests assert neither raw events nor
plaintext passwords reach disk. No regulatory-compliance claim is made.

### 12. Limitations

Real accuracy unmeasured; distributions overlap; browser-side bot detection is an
arms race against an adversary controlling the client; enrollment is
trust-on-first-use; pointer features are modality-dependent; rate limiting is
in-process; no adaptive profiles. Samples are small and every rate is reported
with its *n*.

### 13. Future work

Adaptive profile updates on high-confidence accepts; a larger population prior;
per-modality thresholds; Chrome MV3 packaging of the same collector; digraph-
conditioned keystroke features once enough data exists to estimate them.

---

*140 automated tests, randomised ordering, warnings as errors. All behavioural
logic authored within the event window.*
