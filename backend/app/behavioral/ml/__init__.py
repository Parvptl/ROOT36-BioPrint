"""RETIRED — per-user Isolation Forest anomaly layer.

**This package is not part of the production authentication pipeline.** No
login, enrollment or decision path imports it. It is kept so the experiment
that produced the decision to retire it remains reproducible, and its tests
still run.

Do not reintroduce it into the identity decision.

Why it was built
----------------
The idea was a second, independent opinion on identity: fit a one-class model
per user on their enrollment behaviour, and let "this capture is unusual for
this account" corroborate the statistical fingerprint.

Why it was retired
------------------
**1. It answered the wrong question.** An anomaly detector asks "is this
behaviour unusual?" Authentication needs "is this the enrolled user?" Those
diverge: a genuine user having an unusual day is anomalous but authentic, and
an impostor whose typing happens to sit inside the population's dense region is
unremarkable but wrong. The personalised profile answers the second question
directly, and the automation detector already owns "is this a human at all".

**2. It measured worse, monotonically.** evaluation/hybrid_comparison.py, 25
enrollments, 200 genuine and 200 impostor attempts:

    statistical only (1.0/0.0)   7.8% EER
    hybrid           (0.8/0.2)   9.5%
    hybrid           (0.6/0.4)  12.0%
    ML only          (0.0/1.0)  15.5%

Every weight including the ML term was worse than excluding it. The diagnosis
is in docs/ml-layer.md: the anomaly score has a larger raw separation but a
five times larger spread on genuine users, and it correlates positively with
the statistical score — so it compounds that layer's mistakes instead of
catching them. It shipped at weight 0.0 for that reason.

**3. It could not change any verdict, and cost ~30.7 ms p50 to not do so.**
evaluation/ml_independence.py ran 60 scenarios — enrollment, genuine, human
impostor, scripted bot, value-injection bot, wrong password, nonce reuse, stale
capture, adaptation trail, final profile state — at both 2-capture and
8-capture enrollment, with the model enabled and bypassed. **Zero differences.**
A component that provably cannot affect the outcome is not a security layer; it
is latency.

What replaced it
----------------
Nothing. The layers that remain each answer a question it did not:

    credential          Argon2id
    replay / integrity  single-use nonce, randomised phrase, capture age
    automation          bot detector — "is this a human?"
    identity            personalised behavioural profile — "is this YOU?"

The Aalto corpus stays a **population prior** — per-feature variability used to
scale a personal profile. It is not an identity model and must not become one.
"""
