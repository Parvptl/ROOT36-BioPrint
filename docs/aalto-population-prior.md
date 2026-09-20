# The Aalto population prior

What it is, how it is generated, and what it deliberately does not contain.

## What the prior is for

One question, per keyboard feature: **how much do humans in general differ on
this?** That is what makes a deviation meaningful. Typing 15 ms slower than
usual is unremarkable if everyone's dwell times span 60 ms; it is significant
if they span 8 ms.

    Aalto  =  "how much do people vary?"
    User   =  "what does THIS person do?"

The prior is **not identity-training data**. No authentication classifier is
trained from Aalto, no participant id is ever a class, and nothing in the
pipeline learns to recognise an Aalto participant. The dataset contributes
exactly one thing: a per-feature robust scale, which `PopulationPrior.scale_for`
uses as a floor and `_discriminability_weight` uses as the denominator of a
Fisher-style ratio.

    Aalto population statistics
        → population variability prior
        → new BioPrint user enrollment
        → personalised user profile
        → behavioural authentication

## Artifacts

| path | what it is |
|---|---|
| `backend/data/aalto_prior_real.json` | **ACTIVE**. Generated from the full Aalto corpus. |
| `backend/data/aalto_prior_synthetic_bootstrap.json` | preserved historical baseline |
| `backend/data/aalto_prior.json` | the former production artifact, byte-identical to the bootstrap; kept for rollback |

The real prior was activated after a controlled A/B in which the only variable
was which artifact loaded. At two captures and the same threshold, false
acceptance fell from 29.8% to 10.1%; equal-error rate and ROC-AUC improved at
every enrollment size; cold-start false rejection rose from ~0.7% to ~3.2%.
Generated typists — not a real-human study.

Rollback is one environment variable:

```bash
BIOPRINT_AALTO_PRIOR_PATH=data/aalto_prior.json
```

## Reproducing the real prior

```bash
cd backend && ./.venv/Scripts/python.exe -m app.behavioral.fingerprint.precompute_aalto_prior --data-dir evaluation/ml/data/aalto/extracted/Keystrokes/files --output data/aalto_prior_real.json
```

The script **refuses to overwrite an existing output file**. Move the old one
aside first; a prior that silently replaced its predecessor would destroy the
only record of what a previous evaluation was measured against.

- **Input:** `backend/evaluation/ml/data/aalto/extracted/Keystrokes/files/`,
  one `*_keystrokes.txt` TSV per participant.
- **Output:** `backend/data/aalto_prior_real.json`.
- **Raw dataset is gitignored** and must never be committed.

### The pipeline, end to end

| stage | where | what happens |
|---|---|---|
| parse | `evaluation/ml/aalto_adapter.py::iter_sessions` | streaming stdlib TSV reader, splits on `TEST_SECTION_ID` |
| convert | `aalto_adapter._to_keypress` | JS keycode → DOM `code`, → `KeyPress(ctx="phrase")` |
| extract | `app/behavioral/features/keyboard.py::extract` | **the same function the browser pipeline calls** |
| gate | `aalto_adapter.extract_session_features` | enabled, finite, `count >= min_observations` — the same three gates as `features/extractor.py` |
| per participant | `precompute_aalto_prior.aggregate_user_features` | median across that participant's sentences; a feature seen in fewer than 2 sentences is dropped |
| population | `precompute_aalto_prior.compute_population_stats` | median, MAD, `robust_scale` = MAD × 1.4826, percentiles; features with fewer than 10 participants are omitted |
| serialize | `precompute_aalto_prior.save_prior` | JSON with metadata, processing report, input manifest |
| load | `app/behavioral/fingerprint/aalto_prior.py::load_aalto_prior` | → `AaltoPrior` → `PopulationPrior` |

**No keyboard feature mathematics is implemented anywhere in this path.** The
adapter builds the same structures the browser builds and hands them to the
same extractor. If a dwell or a flight time were ever computed inside the
adapter, the prior would stop describing the shipped system — which was the
exact objection that ruled out the CMU and KeyRecs datasets, documented in
[datasets.md](datasets.md).

### Participant balancing

Aalto participants contributed different numbers of sentences. Pooling
sentences would let a participant with 30 sections count thirty times as much
as one with 5 toward "how much do humans vary".

So sentences collapse to **one feature vector per participant** before any
population statistic is computed. `n_users` in the artifact is therefore a
participant count, not a sentence count, and each participant contributes at
most one observation to each feature.

### Reproducibility

The artifact records its own provenance: `adapter_version`, `generated_at`,
the full `statistics_method` block, the processing report (files discovered,
participants processed and failed, sessions discovered, extracted and rejected,
rejection reasons), and a `manifest` — a SHA-256 over the sorted
`filename:bytes` lines of the input directory. The manifest establishes later
that the same corpus was processed without hashing 168k file bodies.

## The unobserved feature

`kbd_shift_right_ratio` measures which Shift key a person reaches for. Aalto
records **keycode 16 for Shift and does not say which one**, so the trait this
feature measures is not present in the data.

It is therefore recorded as:

```json
"kbd_shift_right_ratio": {
  "status": "unobserved",
  "reason": "Aalto records keycode 16 for Shift without distinguishing ...",
  "n_users": 0
}
```

with **no median, no MAD, no scale and no percentiles**. Not inferred, not
imputed to a mean, not set to zero, not randomly assigned.

### Why the representation matters

Two things must never be confused:

| | meaning | correct behaviour |
|---|---|---|
| unobserved | we have no idea how much people vary | fall back to the relative prior |
| `scale == 0.0` | people do not vary at all | any deviation is infinitely significant |

Three independent mechanisms keep them apart:

1. **Encoding.** Keycode 16 maps to `key_class="shift"`, which matches neither
   `shift_left` nor `shift_right` in `keyboard._shift_features`. The shift
   count is zero, so the extractor **never emits the feature at all**. Nothing
   downstream has to remember to remove a fabricated value, because none is
   ever created. Mapping keycode 16 to `ShiftLeft` — as an earlier version of
   the precompute script did — would have produced a ratio of exactly 0.0.
2. **Type.** `UnobservedFeature` has a name and a reason and no numeric fields.
   The loader puts unobserved entries there rather than in `AaltoFeatureStat`,
   so there is no median or scale to misread. Without this branch,
   `float(data.get("population_scale", 0.0))` would have turned the JSON above
   into a zero-median, zero-scale entry indistinguishable from a real
   measurement.
3. **Fallback.** Unobserved features contribute no entry to
   `PopulationPrior.scales`, so `scale_for` returns
   `max(RELATIVE_SPREAD × |own_median|, ABSOLUTE_FLOOR)` — the same path a
   pointer feature takes. `to_population_prior` additionally filters
   `scale > 0.0`, and `scale_for` guards `empirical <= 0.0` again.

`tests/test_aalto_prior.py::test_an_unobserved_feature_never_becomes_zero_variability`
is the load-bearing test.

## What Aalto cannot provide at all

- **7 pointer features** — Aalto has no mouse data.
- **4 interaction features** — Aalto has no form events.

Both modalities fall back to the relative prior. No numbers are fabricated for
them, and the loader actively rejects any prior file that claims to contain
them.

Aalto also captured **desktop transcription typing**, not browser form-filling
against a randomised challenge phrase. The population spread of a timing
feature is a reasonable thing to carry across that gap; a population *median*
is less so, which is one reason only the scale is consumed downstream.
