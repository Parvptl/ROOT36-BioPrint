# Live demo runbook

Target: **4 minutes**. Covers everything the problem statement asks a demo to
show — genuine enrollment and login, a correct-password impostor, and a
scripted attempt, both detected and blocked.

---

## Before the judges arrive

### 1. Enroll for real (~4 minutes, do this well ahead)

**This cannot be faked and must not be.** The demo account has to be enrolled by
the person who will demo, typing normally. A profile built any other way will
not match them at the keyboard and the demo will fail in front of the panel.

Build the frontend once, then run the backend — it serves the built app, so
everything is on one port and there is no CORS to get wrong live.

```bash
cd frontend && npm run build
cd ../backend && ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Check <http://127.0.0.1:8000/health> shows `"frontend_bundled": true`.

Go to <http://127.0.0.1:8000/enroll>, register, complete all **8 rounds**. Type
the way you normally type — do not be careful or neat. An artificially tidy
baseline rejects the real you later.

### 2. Have the impostor practise once

Your teammate should do **one** login attempt with your password before the
demo, so they are not fumbling with an unfamiliar form live. Their failed
attempt is fine; it stays in the audit trail and is honest.

### 3. Warm the server up

The first request after startup pays import and allocation costs and can read
~120 ms instead of ~55 ms. Do one throwaway login before presenting so the
latency on screen is representative.

### 4. Open two windows

- **Terminal**: in `backend/`, ready to run the attack scripts

### 5. Reset only if you must

```bash
cd backend && ./.venv/Scripts/python.exe -m app.demo_reset
```

This deletes users, profiles, challenges and audit rows. It plants nothing and
forces no decision. **It also deletes your enrolled profile**, so only run it if
you have time to enroll again.

---

## The script

### 1 · The problem — 20s

> "Passwords leak. Once an attacker has yours, a normal login has nothing left
> to check — the credential matched, so the door opens. The usual fix is an OTP,
> which this challenge rules out. BioPrint adds a factor the attacker can't
> steal with the password: how you actually type and move."

### 2 · Enrollment — 25s

Show `/enroll`. Do **one round only**, narrating:

> "Eight rounds, about three minutes, one time. Each one asks for a different
> random phrase — so we learn how you type, not what you typed. That matters:
> the fingerprint is independent of the password and survives a password change."

Then switch to the already-enrolled account.

### 3 · Genuine login — 30s

At `/login`, log in normally. **→ ACCESS GRANTED**

> "Password plus behaviour. Both matched."

⚠️ **If this blocks, do not panic and do not hide it.** Say:

> "That's a false rejection — we measure about 7%. This is a probabilistic
> signal and we report it as one; the alternative is a threshold so loose it
> accepts impostors. Let me retry."

Then retry. Handled that way it reads as rigour, not failure. A team claiming
100% accuracy on a biometric is the one a judge should distrust.

### 4 · The impostor — 45s ← **the central claim**

Hand the keyboard to your teammate. They type **your correct password**, out
loud so the panel hears it.

**→ ACCESS BLOCKED — BEHAVIORAL MISMATCH**

> "Correct password. Different person. Blocked — no OTP, no second channel,
> purely on how they typed."

Point at the signal bands:

> "It tells them *which category* disagreed — typing rhythm, pointer, form
> interaction. It does **not** tell them the score or the threshold. That was
> deliberate: an earlier version returned both, and that let an attacker read
> their exact distance from acceptance and hill-climb toward it. We took it out."

### 5 · The bot — 35s

```bash
./.venv/Scripts/python.exe ../demo/attack_bot.py --username <you> --password <yours>
```

**→ AUTOMATION DETECTED**

> "Same correct password, driven by a script. Reported as automation, not as a
> behavioural mismatch — 'this isn't you' and 'this isn't a person' are
> different findings and the system distinguishes them."

If asked *"couldn't you just check `navigator.webdriver`?"*:

> "That's one line to delete, and CDP-driven input arrives with `isTrusted` set
> to true. We weight both at 0.35 and there are tests proving neither can block
> on its own. We caught a real CDP browser attacking this page — `isTrusted`
> true, `webdriver` false — on event ordering. It's in the repo as a regression
> test."

### 6 · Replay — 30s

```bash
./.venv/Scripts/python.exe ../demo/attack_replay.py --username <you> --password <yours>
```

Three beats on screen: accepted → `CHALLENGE_REUSED` → `PHRASE_MISMATCH`.

> "Record a genuine login and play it back. The nonce is single-use, so the
> straight replay dies. And the phrase is regenerated every attempt, so the
> recording answers the wrong question."

### 7 · The console — 40s

Switch to `/security`.

> "This is the operator view — the numbers the login page withholds. Every
> attempt, its verdict, the identity deviation, the automation score, latency."

Point at the ledger: genuine ~0.08, impostor higher, bots at 0.99, replays
failing on integrity. Then:

> "Behavioural analysis is about 2 milliseconds, p95 2.4. End to end is ~55,
> and 95% of that is Argon2 doing its job on the password."

### 8 · Why it holds up — 25s

> "Three things. The browser is only a sensor — it ships raw events and never a
> score, so patching our JavaScript doesn't move the decision. Every feature
> declares a noise floor, because we measured one feature contributing 23% of a
> genuine user's deviation from pure counting noise. And the threshold is
> calibrated from leave-one-out data, not picked — where we don't have enough
> data to calibrate, the profile says so."

---

## Questions to expect

**"What's your accuracy?"**
> "On synthetic typists, roughly 8% equal-error across 40 enrollments. Against
> real people it's pending — that needs humans at a keyboard, and we won't
> invent the number. Everything in the README is labelled which kind it is."

**"Could a determined attacker beat it?"**
> "Yes. Someone driving a real browser with timings sampled from real humans
> isn't defeated by browser-side detection. We raise cost from trivial to
> substantial. We don't claim impossibility — that claim would be false."

**"Why not deep learning?"**
> "Eight enrollment samples in twenty-seven dimensions. A one-class SVM or an
> autoencoder isn't estimable there — it fits enrollment noise and produces
> confident nonsense. We fit them in the evaluation harness so the choice is
> measured, not assumed."

**"Isn't the phrase just a CAPTCHA?"**
> "It's seven ordinary words and takes four seconds. It's doing two jobs: it
> makes replay fail, and because it changes every time, the features describe
> how you type rather than what you typed — which is what keeps the fingerprint
> independent of the password."

---

## If something breaks

| Symptom | Do this |
|---|---|
| Genuine login blocked | Retry, and explain the 7% FRR. Don't hide it |
| "Cannot reach the service" | Backend down, or the bundle was served from another port. Restart the backend |
| Attack script errors | Run from `backend/`, not the repo root |
| Everything is wrong | `python -m app.demo_reset`, then re-enroll. **Needs ~4 min** |

Never let a technical error read as an authentication success. The system is
built so a backend failure surfaces as an error, never as ACCESS GRANTED.
