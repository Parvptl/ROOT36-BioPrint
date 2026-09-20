# Phase E: real-human collection protocol

Phase D measured the real Aalto prior against the synthetic bootstrap on
*generated typists* and found consistent differences. It could not say whether
those transfer to real browser users, and it could not be made to: BioPrint
discards raw events at extraction and deletes enrollment vectors once a profile
is fitted, so the feature vectors behind real login attempts do not exist.

This phase adds the minimum machinery to collect them — **opt-in, off by
default** — so one real human session can be scored under both priors.

## What is stored, and what is not

Evaluation capture writes to a **separate database**
(`backend/data/evaluation/eval_sessions.db`), never to the product database.
Privacy verification is therefore a check on one file.

**Stored**

| field | why |
|---|---|
| `participant` | pseudonymous id of whoever was typing |
| `target` | pseudonymous id of the account being claimed |
| `session_id` | random uuid4, for deduplication |
| `role` | `enrollment` or `login` |
| `label` | `genuine` or `human_impostor` |
| `enrollment_index` | ordinal, so rounds stay in order |
| `features_json` | the derived feature vector |
| `coverage` | fraction of the registry observed |
| `created_at` | unix time |

**Never stored**

passwords · password hashes · any password-derived value · raw keystroke events
· raw pointer trajectories · the challenge phrase or any typed text · usernames
· emails · names · IP addresses · user agents

The features written are the same derived numbers that already reach the
profile. **This is not a raw-event retention switch — no such switch exists.**
Events live inside one function call, are extracted, and are dropped. The
product privacy model is unchanged; this is an evaluation harness bolted
beside it.

`app/evaluation_capture.py` carries a `FORBIDDEN_KEYS` set and refuses any
vector containing one, so a future feature literally named `password` cannot
reach the dataset by accident.

## Pseudonyms

The participant id is an HMAC of the username under the server secret,
truncated: `alice → E08af799a15`. A bare hash of a short username is reversible
by guessing, which would put the name back into the dataset in all but name.

**No table maps a pseudonym back to a person.** The operator keeps that mapping
outside the dataset; `eval_collect.py pseudonym <name>` recomputes it on demand.

Because the id is keyed on the secret, collection **refuses to run** when
`BIOPRINT_SECRET_KEY` is unset. With an ephemeral secret a restart halfway
through would give every participant a new id and split one person's rounds
across two identities that can never be rejoined — silently. Set the secret in
`backend/.env` first.

## Labelling

The server cannot know who is at the keyboard, so the **operator declares it**
between phases. The collector compares the declaration to the account being
claimed:

    actor == target  →  genuine
    actor != target  →  human_impostor

The label never comes from the verdict. Deriving it from the outcome would
define an impostor as "an attempt the system rejected", which is the thing
under test.

An enrollment round submitted while a non-owner is declared is **refused**: a
third party's rounds must never become someone's baseline.

## Operator workflow

**Once**, before any participant:

```bash
cd E:/ROOT36/backend && grep -q BIOPRINT_SECRET_KEY .env && echo "secret set" || echo "SET BIOPRINT_SECRET_KEY IN .env FIRST"
```

Start the backend with capture on:

```bash
cd E:/ROOT36/backend && BIOPRINT_EVALUATION_MODE=1 ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

**Per participant** (target 10–15 participants; 10 is the minimum the analysis
will report on):

1. Register an account for them in the UI at `http://127.0.0.1:8000`.
2. Declare them as the actor:

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/eval_collect.py actor <their-username>
```

3. They complete **8 enrollment rounds**, typing naturally. Do not coach their
   typing — an instructed typing style is not their typing style.
4. They complete **5–10 genuine logins**, entering the correct password and
   typing the challenge naturally.

**Impostor phase.** Declare the person who is actually typing, then have them
log in against a *different* participant's account using that account's test
credential:

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/eval_collect.py actor <impostor-username>
```

They should type **as themselves**. Do not ask them to imitate the target
unless a deliberate impersonation condition is being run — that is a different
experiment with a different threat model.

**Between participants**, clear the actor so stray traffic is not mislabelled:

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/eval_collect.py actor --clear
```

**Check progress at any time:**

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/eval_collect.py status
```

## Analysis

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/eval_collect.py validate
```

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/real_human_ab.py
```

`validate` enforces dimensionality, finiteness, unique session ids, contiguous
enrollment ordinals and label consistency, and prints a reason per rejection
rather than discarding quietly.

`real_human_ab.py` scores **the same collected vectors** under both priors,
fitting a fresh independent profile per condition. It **refuses to report a
rate** below 10 usable participants and says so instead.

## Verifying the plumbing without participants

```bash
cd E:/ROOT36/backend && ./.venv/Scripts/python.exe evaluation/rehearse_collection.py
```

Drives the real HTTP API with generated typists, confirms capture and
labelling, scans the dataset file for leaked strings, then **deletes the
dataset** — generated typists must never be mistaken for real-human evidence.

## After collection

The dataset contains derived features only and is gitignored. When the
experiment is finished, delete `backend/data/evaluation/` — it is regenerable
only by asking people to type again, so archive it deliberately if you intend
to keep it.
