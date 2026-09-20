# Public behavioural-biometric datasets: evaluation for BioPrint

Researched 2026-09-20, hour 12.4 of 36. Sources are the original publishers, not
Kaggle or GitHub mirrors.

**Headline finding: the most-cited dataset in this field is not suitable for this
system, and the reason is architectural rather than practical.**

---

## The compatibility constraint that drives everything

BioPrint is deliberately **password-independent and free-text**. Every login
issues a *randomly generated* phrase, and identity features are computed only
over that phrase. This is not incidental — it is what makes the fingerprint
survive a password change, and it is a stated requirement of the challenge
("a behavioural fingerprinting engine ... independent of their password").

Our features are therefore **aggregate timing statistics over arbitrary text**:
dwell median/MAD, flight median/IQR, negative-flight fraction, log inter-key
median/MAD, burst fraction, pause rate, correction behaviour, shift-hand
preference.

A dataset is usable only if it provides **raw press/release timings over varied
text**. A dataset of per-digraph timings for one fixed string is measuring a
different problem.

---

## Comparison

| Dataset | Modality | Users | Sessions / samples | Genuine vs impostor | Licence | Verdict |
|---|---|---|---|---|---|---|
| **CMU Keystroke Benchmark** | keystroke | 51 | 400 reps × 8 sessions | impostor = other subjects | Public download, cite DSN-2009 | **Reject — fixed-text** |
| **Balabit Mouse Challenge** | mouse | 10 | few remote sessions each | test sets mix in other users | No explicit licence; citation requested | **Reject — context mismatch** |
| **KeyRecs** | keystroke | 99 | fixed-text + **free-text** | impostor = other subjects | Zenodo; *Data in Brief* typically CC BY (verify on record) | **Only viable candidate** |

---

## CMU Keystroke Dynamics Benchmark — rejected

**Source:** <https://www.cs.cmu.edu/~keystroke/> (Killourhy & Maxion, DSN 2009).

51 subjects each typed **the same password, `.tie5Roanl`, 400 times** across 8
sessions. Timestamps accurate to ±200 µs. The published benchmark reports EER
between 9.6% and 10.2% across 14 algorithms.

**Why we reject it.** Every sample is 11 keystrokes of *one fixed string*. The
feature space is per-key hold and per-digraph latency for that specific string.
Our system never sees a fixed string — it sees a different random phrase every
attempt, and its features are distribution statistics over whatever was typed.

Validating on CMU would mean building a second, fixed-text feature extractor
that the product does not use, measuring *that*, and reporting the number as
evidence for BioPrint. It would be a real EER on a **different system**. That is
precisely the mismatch the project brief warns against, and it would be
misleading to a judge.

It remains an excellent dataset — for fixed-text keystroke dynamics, which is
not what we built.

## Balabit Mouse Dynamics Challenge — rejected

**Source:** <https://github.com/balabit/Mouse-Dynamics-Challenge> (Fülöp,
Kovács, Kurics, Windhager-Pokol, 2016).

10 user accounts, mouse traffic captured by a **network monitoring device
inspecting RDP** between a client and a remote desktop. Test sessions
artificially mix in other users' data to simulate illegal use.

**Why we reject it.** Three reasons, in order of severity:

1. **Capture context differs fundamentally.** RDP-transported pointer traffic
   through a network monitor is not browser `pointermove` events on a login
   form. Sampling rate, event coalescing and coordinate space all differ.
   Velocity, tremor and straightness computed over RDP traffic are not the same
   measurements our extractor produces.
2. **10 users.** Our own synthetic population already has 6, and our real
   evaluation will have more. Ten is not a meaningful population prior.
3. **No explicit licence** is stated on the repository; only a citation request.
   For a judged submission we prefer an unambiguous licence.

Mouse is also our *weakest* modality by design — 7 of 27 features, mostly absent
during typing — so this would validate the least load-bearing part of the system.

## KeyRecs — the only viable candidate

**Source:** *Data in Brief* 50:109509 (2023), DOI
`10.1016/j.dib.2023.109509`. Data on Zenodo, DOI `10.5281/zenodo.7886743`.
Dias, Vitorino, Maia, Sousa, Praça — GECAD, ISEP/IPP Porto.

**99 participants**, recruited via Prolific, completing **both** a password
retype exercise (fixed-text) and a **text transcription exercise (free-text)**,
plus demographics.

**Why it is the only candidate.** The free-text transcription portion is the
same *kind* of task our challenge phrase is: arbitrary text, typed naturally,
with impostors being other subjects. If it exposes raw press/release timings, our
existing extractor can consume it with only a format adapter — no second feature
implementation, which is the condition that made CMU unusable.

### RESOLVED — incompatible. No loader written.

The open question was whether the release ships raw keydown/keyup events or only
precomputed digraph latencies. Checked against the Zenodo record (v1.0.0,
published 2 May 2023) and the *Data in Brief* descriptor:

> The samples consist of inter-key latencies computed by measuring the time
> between each key press and release during an exercise, following a **digraph
> model**.

**KeyRecs ships precomputed digraph latency features, not a keystroke event
log.** There are no per-event absolute timestamps and no press/release columns.
(For contrast, Clarkson University Keystroke Dataset II *does* ship this — three
columns: timestamp, press/release flag, key code.)

What that costs us:

| our feature | computable from digraph latencies? |
|---|---|
| `kbd_dwell_median`, `kbd_dwell_mad` | **No** — needs per-key hold = keyup − keydown |
| `kbd_dwell_left_median`, `kbd_dwell_right_median` | **No** — same, plus per-key identity |
| `kbd_flight_negative_frac` | **No** — needs keyup[i] against keydown[i+1] |
| `kbd_logiki_median/mad`, `kbd_burst_fraction`, `kbd_pause_rate` | Partially, depending on which digraph timing variant was stored |

Four of our sixteen keyboard features are **uncomputable**, and they include the
dwell statistics that carry high weight in the profile. Negative-flight fraction,
one of our most discriminative features, is also lost.

Consuming KeyRecs would therefore mean writing a **second, reduced feature
extractor** the product does not contain and reporting its performance — exactly
the objection that ruled out CMU. Per the project decision, we stopped here
rather than forcing it.

**Outcome: no public-dataset validation.** Documented, not worked around.

---

## REAL AALTO DATASET

**Source:** Aalto University Keystroke Dataset (136M Keystrokes).
**Status:** Available locally at `backend/evaluation/ml/data/aalto/extracted/Keystrokes/files/`

This dataset contains raw, chronological keystroke events with absolute millisecond timestamps (`PRESS_TIME`, `RELEASE_TIME`) and keycodes over free-text transcription tasks. It provides exactly what BioPrint needs to compute 15 of its 16 keyboard features.

### Feature Compatibility and Adapter Status

An adapter (`backend/evaluation/ml/aalto_adapter.py`) maps the Aalto data into the exact `KeyPress` objects expected by the BioPrint production browser extractor, guaranteeing mathematical equivalence.

**Compatible Features (15/16):**
All timing-based and rhythm features (e.g., `kbd_dwell_median`, `kbd_flight_negative_frac`, `kbd_logiki_median`) map perfectly because Aalto provides high-precision press and release times. `KEYCODE` values are deterministically mapped to left/right hands.

**Incompatible Feature (1/16):**
- `kbd_shift_right_ratio`: **UNOBSERVED**. Aalto logs generic `KEYCODE = 16` for all Shift keys, lacking modern `event.code` or `event.location` data to distinguish Left vs Right Shift. This feature is explicitly marked as `NaN` (unobserved) in the Aalto adapter rather than being fabricated. The production schema remains unchanged.

### Demographics and Identity
- **Participant/Session:** `PARTICIPANT_ID` maps to the user identity. `TEST_SECTION_ID` corresponds to a single sentence typing session. 
- **Usage:** This dataset is exclusively for calculating population-level variability priors (median/MAD statistics). It is **not** used to learn user identities or replace the 1-sample enrollment lifecycle.

*Note: The real Aalto prior will replace the synthetic bootstrap prior once population statistics are computed in Phase C.*

---

## Outcome

**Public Dataset Validation:** We are actively using the **Real Aalto Dataset** for population prior generation.

| | why rejected / accepted |
|---|---|
| CMU | Rejected: fixed-text; one password string; validates a different architecture |
| Balabit | Rejected: RDP-captured mouse, 10 users, no explicit licence |
| KeyRecs | Rejected: digraph latencies only; no per-key hold times → 4 of 16 keyboard features uncomputable |
| Aalto (136M) | **Accepted**: Raw keydown/keyup events over free text. 15 of 16 features perfectly compatible. |
compatibility, each for a different reason:

| | why rejected |
|---|---|
| CMU | fixed-text; one password string; validates a different architecture |
| Balabit | RDP-captured mouse, 10 users, no explicit licence |
| KeyRecs | digraph latencies only; no per-key hold times → 4 of 16 keyboard features uncomputable |

A dataset with **raw keydown/keyup events over free text** would be usable
directly. Clarkson University Keystroke Dataset II appears to have that shape and
is the obvious next candidate if this work continues past the hackathon; it was
not pursued here because it requires an access request rather than a direct
download.

This is a defensible position, because:

- The system is deliberately free-text and content-independent, and free-text
  behavioural datasets with raw browser-equivalent timings are genuinely scarce.
- We already have a reproducible synthetic benchmark that isolates algorithm
  behaviour under controlled conditions, clearly labelled as synthetic.
- The evaluation that actually matters for this product — real people using
  *this* interface — is collected through our own labelled harness.

**What we will not do:** report a number obtained on a fixed-text dataset, using
a feature extractor the product does not contain, as evidence of BioPrint's
accuracy. A borrowed benchmark figure that does not describe the shipped system
is worse than no figure.

---

## Sources

- CMU Keystroke Dynamics Benchmark — <https://www.cs.cmu.edu/~keystroke/>
- Killourhy & Maxion, *Comparing Anomaly-Detection Algorithms for Keystroke
  Dynamics*, DSN 2009, pp. 125–134
- Balabit Mouse Dynamics Challenge — <https://github.com/balabit/Mouse-Dynamics-Challenge>
- Fülöp, Kovács, Kurics, Windhager-Pokol (2016), Balabit Mouse Dynamics
  Challenge data set
- KeyRecs — <https://www.sciencedirect.com/science/article/pii/S2352340923006091>,
  data at <https://doi.org/10.5281/zenodo.7886743>
