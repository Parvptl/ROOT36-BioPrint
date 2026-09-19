# Run 1 real-human evaluation — forensic analysis

Collected 2026-09-20, 00:57–01:09. Account `parvptl`, threshold 0.3280
(`genuine_only`, no population prior available at enrollment time).

Everything below comes from the preserved audit trail at
`evaluation/out/baseline/run1_2026-09-20_bioprint.db`. Values the system did
not record are marked NOT RECORDED and are not estimated.

---

## Two findings that come before any accuracy number

### 1. The run measured a build that is not the product

The uvicorn process serving this evaluation started **2026-09-19 20:30:44**. It
therefore predates:

| commit | time | layer |
|---|---|---|
| `0721e3e` | 2026-09-19 21:45 | per-user Isolation Forest anomaly layer |
| `33b568b` | 2026-09-20 00:40 | confidence-gated adaptive profiles |

Three independent confirmations in the database:

- `profile_updates` contains **zero rows**, across 16 ALLOW decisions, five of
  which qualify as HIGH confidence under the shipped policy;
- `backend/data/models/` **does not exist** — no model was trained at either
  enrollment;
- `ml_anomaly_score` and `statistical_identity_score` are **NULL on all 62
  attempts**.

Nothing in the run's own output revealed this. The database *schema* was
current, because the file had been recreated by a separate process after the
schema change, while the long-lived server kept serving the code it loaded
hours earlier.

**Consequence:** run 1 is a valid measurement of the statistical identity layer
plus bot detection and replay defence. It says nothing about the ML layer or the
adaptive layer, which were never exercised.

A preflight check now refuses to run an evaluation against a server older than
the source tree (`evaluation/run_evaluation.py::preflight`, backed by
`started_at` on `/health`).

### 2. Password failures were counted as behavioural decisions

Three of the 43 labelled attempts were rejected by the credential gate, which
runs **before** any behavioural analysis. They have no identity score, no
threshold comparison and no bot score. Counting them inflated FRR with typing
accuracy and inflated impostor rejection with a credential check the
behavioural engine never contributed to.

| | as reported in run 1 | corrected |
|---|---|---|
| Genuine acceptance | 85.7% (12/14) | **92.3% (12/13)** |
| Genuine FRR | 14.3% | **7.7% (1/13)** |
| Impostor rejection | 100% (10/10) | **100% (8/8)** |
| FAR | 0% | **0%** |
| PASSWORD_FAILURE | not reported | **3** (ids 8, 29, 30) |

Bot (12/12) and replay (7/7) are unaffected; both phases used a correct
password throughout.

---

## The 14 genuine attempts

Phase boundaries were reconstructed from timestamps and cross-checked against
run 1's own aggregate counts; all four phases match exactly. Run 1 predates the
harness writing per-attempt labels — it now does.

`threshold` is the **effective** threshold: the profile threshold tightened by
that attempt's own automation score
(`risk_engine.AUTOMATION_TIGHTENING = 0.5`). It was not stored in run 1 and is
reconstructed here from two recorded values and one constant; live runs now
persist it.

| id | time | decision | reason | identity | threshold | margin | automation | coverage | confidence | adapted | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 00:57:32 | BLOCK | INVALID_CREDENTIALS | — | — | — | — | — | BLOCKED | no row | 111.9 |
| 9 | 00:57:58 | ALLOW | BEHAVIOR_MATCH | 0.0950 | 0.3280 | +0.2330 | 0.0000 | 1.000 | HIGH | **no row** | 121.9 |
| 10 | 00:58:24 | ALLOW | BEHAVIOR_MATCH | 0.1776 | 0.2879 | +0.1103 | 0.2449 | 1.000 | MEDIUM | no row | 122.9 |
| 11 | 00:58:49 | ALLOW | BEHAVIOR_MATCH | 0.1030 | 0.3280 | +0.2250 | 0.0000 | 1.000 | HIGH | **no row** | 125.1 |
| 12 | 00:59:17 | ALLOW | BEHAVIOR_MATCH | 0.1943 | 0.2733 | +0.0790 | 0.3337 | 1.000 | MEDIUM | no row | 121.3 |
| 13 | 00:59:39 | ALLOW | BEHAVIOR_MATCH | 0.1620 | 0.2660 | +0.1041 | 0.3781 | 1.000 | MEDIUM | no row | 122.7 |
| 14 | 01:00:04 | ALLOW | BEHAVIOR_MATCH | 0.0761 | 0.2731 | +0.1969 | 0.3352 | 1.000 | MEDIUM | no row | 119.4 |
| 15 | 01:00:26 | ALLOW | BEHAVIOR_MATCH | 0.1114 | 0.2733 | +0.1619 | 0.3337 | 1.000 | MEDIUM | no row | 122.0 |
| 16 | 01:00:44 | ALLOW | BEHAVIOR_MATCH | 0.1791 | 0.2875 | +0.1084 | 0.2470 | 1.000 | MEDIUM | no row | 118.7 |
| 17 | 01:01:06 | ALLOW | BEHAVIOR_MATCH | 0.1642 | 0.3280 | +0.1638 | 0.0000 | 1.000 | HIGH | **no row** | 136.4 |
| 18 | 01:01:27 | **BLOCK** | **AUTOMATION_DETECTED** | NOT COMPUTED | 0.2442 | — | **0.5112** | 0.815 | BOT | no row | 123.3 |
| 19 | 01:01:50 | ALLOW | BEHAVIOR_MATCH | 0.1233 | 0.3280 | +0.2047 | 0.0000 | 1.000 | HIGH | **no row** | 120.1 |
| 20 | 01:02:10 | ALLOW | BEHAVIOR_MATCH | 0.0622 | 0.2733 | +0.2111 | 0.3337 | 1.000 | MEDIUM | no row | 118.1 |
| 21 | 01:02:30 | ALLOW | BEHAVIOR_MATCH | 0.1484 | 0.3280 | +0.1796 | 0.0000 | 1.000 | HIGH | **no row** | 129.0 |

Confidence is derived from the recorded identity, automation, coverage and
threshold — exactly the inputs `classify_confidence` takes. "no row" means the
running build had no adaptive layer; the five HIGH rows are updates that the
shipped build would have applied and this one did not.

### NOT RECORDED for any attempt

- **per-modality scores.** `RiskDecision.modality_scores` carries keyboard,
  pointer and interaction scores at decision time, but only the reason code and
  the signal list are persisted. The signal list *is* ordered by descending
  modality score, so the ranking survives and the magnitudes do not.
- **statistical vs ML split** — NULL throughout, see finding 1.
- **profile version per attempt** — no `profile_updates` rows exist.

---

## The two genuine rejections

**Neither was an identity-threshold rejection.** Zero of the 13
password-correct genuine attempts were rejected for scoring above threshold.

### id 8 — INVALID_CREDENTIALS

Password mistyped. No behavioural analysis ran. Excluded from FRR.

### id 18 — AUTOMATION_DETECTED, automation 0.5112 against a 0.50 gate

The bot detector fired on a real person. The only contributing signal was
`POINTER_ANOMALY` (`bot_detection/detector.py`, weight 0.6), which fires on any
of: near-straight pointer path, near-zero fine movement (tremor), or
metronomic sample intervals. Coverage was 0.815 rather than the 1.000 of every
other genuine attempt, consistent with a sparse pointer capture — and a sparse
capture is exactly the condition that makes "no tremor" and "straight path"
look synthetic.

Identity was **never scored**: automation is gate 2 and returns before identity
comparison, by design ("this is not a person" is a different finding from "this
is not you").

Genuine automation scores across the phase:

```
0.0000 x5, 0.2449, 0.2470, 0.3337 x3, 0.3352, 0.3781, 0.5112
```

Six of thirteen genuine attempts scored above 0.30 on a detector whose block
gate is 0.50. The margin is thinner than it should be.

---

## Answers

### 1. Keyboard, mouse, interaction, or fusion?

**None of them — the false rejection came from the automation detector, not the
identity comparison at all.** Specifically from the pointer branch of bot
detection. On the identity side, keyboard was the dominant modality on impostor
blocks (KEYSTROKE_MISMATCH named on 4 of 8) while interaction led the signal
ordering on most genuine ALLOWs — but genuine identity never came close to
failing.

### 2. Genuinely anomalous, or just below threshold?

Neither. Both rejections were **categorical, not marginal**: one never reached
behavioural analysis, the other never reached identity scoring. The closest any
password-correct genuine attempt came to the identity threshold was id 12, at
0.1943 against 0.2733 — a margin of 0.0790, or 29% of the bar.

The one genuinely marginal number in the run is the automation score on id 18:
0.5112 against a 0.50 gate, over by **0.0112**.

### 3. Did adaptive updates improve subsequent genuine acceptance?

**NOT MEASURED.** Zero adaptive updates occurred; the running build had no
adaptive layer. The profile stayed at v1 with `update_count = 0` throughout.
Five genuine attempts (9, 11, 17, 19, 21) met the HIGH-confidence gate and would
have produced updates on the shipped build.

### 4. Was the threshold overly conservative?

**No — the opposite, if anything.**

| | n | min | median | max |
|---|---|---|---|---|
| genuine identity | 12 | 0.0622 | 0.1359 | 0.1943 |
| impostor identity | 8 | 0.3908 | 0.4412 | 0.7002 |

Complete separation, with a gap of **0.1965** between the worst genuine and the
best impostor. The threshold 0.3280 sits 68% of the way across that gap —
nearer the impostor side than the middle. It cost zero false rejections.

### 5. Could a threshold change increase genuine acceptance without unacceptable FAR?

**No, because there is nothing for it to buy.** Genuine acceptance among
password-correct, behaviourally-judged attempts was already 12/12. Raising the
threshold cannot improve a rate that is already 100%, and would only erode the
0.1337 of impostor headroom.

The one lever with measurable headroom is the **automation block gate**, not the
identity threshold. Raising it from 0.50 to 0.55 would have admitted id 18,
taking genuine FRR to 0/13, while every bot attempt in the run scored 0.9675 or
0.9971 — far above either value. But this rests on **one** false rejection and
**twelve** bot attempts, which is not enough to move a security gate on. Not
changed.

### 6. Enough data to justify a threshold change?

**No. Collect more genuine sessions — and collect them against the current
build.**

Three reasons, in order:

1. Run 1 measured a build without the ML or adaptive layers. Any threshold
   tuned on it would be tuned for software that no longer exists.
2. The identity threshold has no demonstrated problem to fix. 12/12 genuine
   accepted, 8/8 impostors rejected, complete separation.
3. The automation gate does have a candidate finding, and n=1 is not a basis
   for moving it. A clean run of 10 genuine and 10 impostor attempts on the
   current build, with attention to whether pointer-sparse captures keep
   producing automation scores above 0.30, is what would settle it.

---

## Not explained

- **Attempt 47**, INVALID_CREDENTIALS at 01:08:09, inside the scripted replay
  window where the password is supplied by the harness on the command line. It
  was not one of the seven attempts run 1 counted as replay. No cause was
  established; recorded here rather than rationalised.
- The **intermittent failure** of
  `test_genuine_only_threshold_still_admits_the_genuine_user` under the full
  suite. Observed once, then passed in the following full runs and 0/40 times
  when its body was repeated in isolation. Pre-existing, unrelated to the
  changes in this pass, and not characterised.
