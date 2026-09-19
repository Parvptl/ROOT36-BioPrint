# Self-assessment against the official criteria

No scores are invented here. For each criterion: what the implementation
actually demonstrates, the evidence a judge could check, where it is weak, and
what could still be fixed in the time left.

Written at **hour 8.5 of 36**, with the build feature-complete and real-human
evaluation outstanding.

---

## Creativity — 10 points
*"Originality of the behavioural signals chosen and how imaginatively they're combined."*

**Demonstrates.** The signal set is chosen rather than scraped together. Three
picks are not the obvious ones: **negative-flight fraction** (whether the next
key goes down before the previous comes up, which separates fluent touch typists
from hunt-and-peck in a way raw speed does not), **shift-hand preference**
sourced deliberately from the public challenge phrase rather than the password,
and **Tab-versus-click navigation**, a settled habit an impostor concentrating on
typing rhythm will not think to imitate.

**Evidence.** `backend/app/behavioral/features/registry.py` — every feature
carries a written justification. `challenge.py` capitalises two random words
specifically so Shift is observable without touching the secret.

**Weakness.** Keyboard and mouse dynamics are the expected answer to this brief.
The originality is in the *handling* — content-independence, noise floors,
sourcing — not in the raw signal list. A judge looking for an exotic modality
(scroll, touch pressure, gyroscope) will not find one.

**Could still do.** Nothing worth the risk. Adding a modality now would be
unmeasured at demo time.

---

## Reliability — 25 points ← **heaviest weight**
*"Consistency of correctly accepting genuine users and rejecting impostors across repeated attempts."*

**Demonstrates.** Reliability was treated as a measurement problem. A sweep over
40 independent enrollments (`evaluation/reliability_sweep.py`) rather than
anecdotes from one profile. It found a real defect: `kbd_backspace_rate` carrying
23% of a genuine user's deviation from pure counting noise, because one backspace
in a fifty-key phrase *is* 0.02 and the MAD collapsed below that. Per-feature
noise floors came from that measurement. Enrollment moved 5→8 rounds on evidence
(FRR 16.2%→7.1%).

It also caught a **prior fix that was actively harmful**: a threshold widened
until tests passed accepted 30 of 30 moderately-different impostors, and only
looked correct because the test fixture impostor was a caricature.

**Evidence.** 143 tests, randomised order, seven consecutive clean runs.
Both flaky tests were fixed by asserting the property the system actually claims
rather than by loosening them.

**Weakness — the significant one.** *Real-human accuracy is unmeasured.* Every
number is synthetic and labelled so. A judge is entitled to say the 25-point
criterion is unevidenced on real people, and they would be right. Synthetic
figures also show genuine and impostor distributions **overlapping** — around 8%
equal error, not a clean separation.

**Could still do — highest-value remaining work.** Enroll, collect ~10 genuine
and ~10 impostor attempts, run `run_evaluation.py`. Roughly 30 minutes and it
converts the heaviest criterion from "unevidenced" to "measured, with n stated."
**This is the single biggest score lever left.**

---

## Ease of Use — 10 points
*"How natural and unobtrusive the enrollment and login experience feels."*

**Demonstrates.** Enrollment is the same form as login, so nothing new is learned
twice, and the instruction is explicitly "do not be careful" rather than
"type consistently." Login adds one seven-word phrase, about four seconds. A
blocked user is told which category disagreed, not just "denied."

**Weakness — a real trade-off, made knowingly.** Enrollment is **8 rounds, ~3
minutes**, raised from 5 *because* 5 was unreliable. That is a deliberate
exchange of Ease of Use (10 points) for Reliability (25). A judge who values
onboarding brevity will mark it down, and the honest answer is that we measured
both and chose.

The phrase field is also unavoidable friction. It is defensible — it carries
replay resistance *and* password-independence — but it is friction.

**Could still do.** A progress indicator showing consistency improving across
rounds would make 8 rounds feel purposeful rather than repetitive. ~40 minutes,
low risk, meaningful perceived-effort win.

---

## Customer Satisfaction — 10 points
*"Confidence and comfort a user would feel trusting this with their account."*

**Demonstrates.** The privacy story is concrete rather than reassuring noise.
Password keystrokes collapse to a single class — not even character *class*
survives — and **the schema itself rejects** a payload carrying a key code in
password context, so a modified client cannot exfiltrate through that endpoint.
Raw events live inside one function call and are dropped; the database has no
raw-event column and tests assert it. Consent text names exactly what is recorded.

**Weakness.** A ~7% false rejection rate means roughly one login in fourteen
needs a retry. For a real product that is irritating, and no amount of good
explanation fixes it. There is also no account recovery path — a user whose
behaviour drifts (injury, new keyboard) has no route back in but re-enrollment.

**Could still do.** Nothing structural in the time left. Being straight about
the FRR in the demo is worth more than hiding it.

---

## Novelty in Algorithms — 15 points
*"Sophistication and originality of the statistical or ML approach."*

**Demonstrates.** The honest novelty is not the model class — it is a
deliberately simple model made to work at n=8 through four specific mechanisms:
a **shrinkage floor** against scale collapse; a **noise floor** per feature at
its measurement resolution; **Fisher-style discriminability weighting** that goes
uniform when the population prior is uninformative rather than inventing a
ranking; and **saturating contribution** so one odd axis cannot outvote the rest.
The threshold is **derived** by leave-one-session-out, and where data is
insufficient the profile *records that* instead of implying calibration happened.

**Now shown, not argued.** `evaluation/baseline_comparison.py` fits One-Class
SVM, Isolation Forest, Elliptic Envelope and Local Outlier Factor on identical
data and scores identical attempts. Equal-error over 30 trials:

| model | EER |
|---|---|
| **BioPrint (shrinkage robust)** | **7.1%** |
| EllipticEnvelope | 14.8% |
| LocalOutlierFactor | 17.3% |
| IsolationForest | 17.5% |
| OneClassSVM (rbf) | 17.9% |

Roughly half the error of the best baseline — and the comparison is rigged
*against* us, since missing features are filled with the enrollment median
rather than treated as uncovered.

Writing this assessment is what caught the gap: the README already *claimed*
the harness fitted these baselines, and it did not. The claim was made true
rather than softened.

**Weakness.** A judge expecting deep learning still sees medians and MADs. The
answer is now a table rather than an argument, but it is a table on synthetic
typists.

**Could still do.** Nothing. This criterion is now evidenced.

---

## Latency — 10 points
*"Speed of the decision; should feel instant, not like a background batch job."*

**Demonstrates.** Measured, staged, and reported with the awkward part visible.
`evaluation/benchmark_latency.py`, 60 decisions, warm-up discarded:
**behavioural analysis p50 2.09 ms / p95 2.42 ms**; end-to-end p50 52.4 / p95
60.1 ms, of which **Argon2id is 95%**.

An earlier unexplained 162 ms p95 was chased down to warm-up cost rather than
quoted or ignored.

**Weakness.** Network time is excluded and stated as excluded. Over real HTTP the
first cold request reads ~120 ms, which is why the runbook has a warm-up step —
a judge hitting it cold sees the worse number.

**Could still do.** Nothing needed. This criterion is in good shape.

---

## Overall Innovation — 20 points
*"How compelling and demo-ready the solution is as a genuine rethink of login security."*

**Demonstrates.** The architectural claim is real and checkable: the browser is a
sensor that ships raw events and never a score, so patching the client cannot
move the decision. The **login response is not an oracle** — an earlier version
returned the exact identity score and threshold, which let an attacker holding a
correct password read their distance from acceptance and hill-climb; that was
found and removed, with the numbers moved to a key-gated operator console. The
split between *"this is not you"* and *"this is not a person"* runs through the
whole design.

The strongest single artifact is `evaluation/captures/cdp_automation_2026-09-19.json`:
real CDP-driven automation, `isTrusted: true`, `navigator.webdriver: false`,
caught on event ordering at 0.95 — proof the detector does not lean on the flags
everyone checks.

**Weakness.** No Chrome extension. The deliverable permits "embedded in the
website" and the bundle is served from the backend on one origin, so this is
compliant — but a panel that pictured an extension may read it as scope avoided.
The rethink is also *architectural*, which demos less vividly than a flashy
feature.

**Could still do.** Lead the demo with the impostor case, not the architecture.
The 45 seconds where a teammate says the correct password aloud and is still
refused is the most persuasive thing here.

---

## Where the remaining time is best spent

| Priority | Work | Time | Criterion |
|---|---|---|---|
| **1** | Real-human evaluation: enroll, 10 genuine, 10 impostor, run harness | ~30 min | **Reliability (25)** — converts unevidenced to measured |
| 2 | Rehearse the demo twice, end to end, on the demo machine | ~20 min | Innovation (20) |
| 3 | Enrollment progress/consistency feedback | ~40 min | Ease of Use (10) |
| 4 | Push regularly rather than in one batch | ongoing | Rule compliance |

## What would be dishonest to claim

- Any real-human accuracy figure. None has been collected.
- That the behavioural check is unspoofable. It is probabilistic and overlaps.
- That bot detection defeats a determined adversary. It defeats scripts.
- That the system is compliant with any data-protection regime. Not assessed.
