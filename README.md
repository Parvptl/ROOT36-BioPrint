# BioPrint — Behaviour-Based Login Security

ROOT 36 / IAC 8.0, IIT Palakkad — Event 2 (BioPrint).

Authenticates a login by *how* the person interacts with the page, not only by
the credential they supply. A correct password is necessary but not sufficient.

> Full documentation (architecture, algorithm, threat model, evaluation results,
> setup and demo instructions) is written up as the system lands. This file is a
> stub during early development and will be completed before submission.

## Status

Under active development during the 36-hour hackathon window
(12:00 Sat 19 Sep 2026 → 23:59 Sun 20 Sep 2026).

## Architecture (chosen)

```
Browser (sensor)          →  raw behavioural events only, no scoring
        ↓ HTTPS/localhost
FastAPI backend (trust boundary)
        ↓
server-side feature extraction
        ↓
  ┌───────────┬──────────────┬───────────┐
  │ Identity  │  Automation  │ Integrity │
  └───────────┴──────────────┴───────────┘
        ↓
   risk engine  →  ALLOW / BLOCK  (+ reason codes)
```

The browser never computes a score, a feature vector, or a decision. The server
recomputes everything from the raw event stream.

## Stack

- Frontend: React + Vite + TypeScript
- Backend: Python + FastAPI
- Storage: SQLite
- Credentials: Argon2id
- Behavioural processing: NumPy (request path), scikit-learn (offline baselines)

## Quick start

Backend:

```bash
cd backend
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Then `GET http://127.0.0.1:8000/health` should return `{"status":"ok",...}`.
