import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import { useBehaviorCollector } from '../hooks/useBehaviorCollector';
import { ApiError, api } from '../services/api';
import type { Challenge, EnrollmentProgress } from '../types/api';

type Stage = 'register' | 'rounds' | 'done';

/**
 * Enrollment.
 *
 * Each round presents exactly the same form as the login page. That is not
 * laziness: features extracted under one interaction shape are only comparable
 * to features extracted under the same shape. Enrolling on a different form
 * than you authenticate against would bake a systematic offset into every
 * genuine login.
 *
 * The phrase differs every round, so the profile is built across varied text
 * rather than memorising one string.
 */
export default function EnrollPage() {
  const { collector, bind } = useBehaviorCollector();

  const [stage, setStage] = useState<Stage>('register');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [consent, setConsent] = useState(false);

  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [progress, setProgress] = useState<EnrollmentProgress | null>(null);
  const [roundUsername, setRoundUsername] = useState('');
  const [roundPassword, setRoundPassword] = useState('');
  const [phraseTyped, setPhraseTyped] = useState('');

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const usernameFieldRef = useRef<HTMLInputElement | null>(null);

  const roundsTotal = progress?.sessions_required ?? challenge?.rounds_total ?? 5;
  const roundsDone = progress?.sessions_captured ?? 0;

  const requestRound = useCallback(
    async (user: string, pass: string) => {
      setBusy(true);
      setError(null);
      try {
        const next = await api.enrollmentChallenge(user, pass);
        setChallenge(next);
        setRoundUsername('');
        setRoundPassword('');
        setPhraseTyped('');
        // Restart the capture so each round is an independent observation.
        collector.reset();
        usernameFieldRef.current?.focus();
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Could not start the enrollment round.');
      } finally {
        setBusy(false);
      }
    },
    [collector],
  );

  async function handleRegister(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.register(username, password, consent);
      setStage('rounds');
      await requestRound(username, password);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Registration failed.');
      setBusy(false);
    }
  }

  async function handleRoundSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!challenge || busy) return;

    collector.markSubmit();
    const session = collector.snapshot(challenge.nonce, phraseTyped);

    setBusy(true);
    setError(null);
    try {
      const result = await api.submitEnrollmentRound(username, session);
      setProgress(result);

      if (result.profile_built) {
        setStage('done');
        collector.stop();
      } else if (result.accepted) {
        await requestRound(username, password);
      } else {
        // Round rejected (bad phrase, too little signal). Re-issue rather than
        // folding a poor-quality capture into the baseline.
        setError(result.message);
        await requestRound(username, password);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not submit this round.');
      setBusy(false);
    }
  }

  useEffect(() => {
    if (stage === 'rounds' && challenge && !busy) usernameFieldRef.current?.focus();
  }, [stage, challenge, busy]);

  if (stage === 'done') {
    return (
      <div className="panel">
        <h1 className="panel-title">Behavioural profile created</h1>
        <p className="panel-sub">
          BioPrint has built a baseline from {roundsDone} enrollment rounds for{' '}
          <strong>{username}</strong>. Future logins are compared against it.
        </p>
        {progress?.quality ? <QualityReadout quality={progress.quality} /> : null}
        <Link to="/login" className="muted-link">
          Go to the login page &rarr;
        </Link>
      </div>
    );
  }

  if (stage === 'register') {
    return (
      <div className="panel">
        <h1 className="panel-title">Create an account</h1>
        <p className="panel-sub">
          You will set a password, then complete a few short rounds so BioPrint can
          learn how you interact with the page.
        </p>

        {error ? <div className="notice error">{error}</div> : null}

        <form onSubmit={handleRegister}>
          <div className="field">
            <label htmlFor="reg-username">Username</label>
            <input
              id="reg-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              placeholder="3-32 characters"
              required
            />
          </div>

          <div className="field">
            <label htmlFor="reg-password">Password</label>
            <input
              id="reg-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              placeholder="At least 8 characters"
              required
            />
          </div>

          <div className="consent">
            <input
              id="consent"
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
            />
            <label htmlFor="consent" style={{ textTransform: 'none', letterSpacing: 0 }}>
              I agree to BioPrint recording <strong>timing and movement</strong> while I use
              this page: how long keys are held, the gaps between them, pointer motion and
              the order I move between fields. The characters of my password are never
              recorded. Raw events are discarded after analysis; only summary statistics
              are stored.
            </label>
          </div>

          <button className="primary" type="submit" disabled={busy || !consent}>
            {busy ? 'Creating account...' : 'Create account and begin'}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="panel">
      <h1 className="panel-title">
        Enrollment round {Math.min(roundsDone + 1, roundsTotal)} of {roundsTotal}
      </h1>
      <p className="panel-sub">
        Fill this in the way you normally would. Do not try to be careful or
        consistent &mdash; BioPrint is learning your ordinary behaviour, and an
        artificially neat baseline would reject the real you later.
      </p>

      <div className="rounds">
        {Array.from({ length: roundsTotal }, (_, i) => (
          <div
            key={i}
            className={`round-pip ${i < roundsDone ? 'done' : i === roundsDone ? 'active' : ''}`}
          />
        ))}
      </div>

      {error ? <div className="notice warn">{error}</div> : null}

      <form onSubmit={handleRoundSubmit}>
        <div className="field">
          <label htmlFor="enr-username">Username</label>
          <input
            id="enr-username"
            ref={(el) => {
              usernameFieldRef.current = el;
              bind('username')(el);
            }}
            value={roundUsername}
            onChange={(e) => setRoundUsername(e.target.value)}
            autoComplete="off"
            required
          />
        </div>

        <div className="field">
          <label htmlFor="enr-password">Password</label>
          <input
            id="enr-password"
            ref={bind('password')}
            type="password"
            value={roundPassword}
            onChange={(e) => setRoundPassword(e.target.value)}
            autoComplete="off"
            required
          />
        </div>

        <div className="phrase-card">
          <div className="phrase-label">Type this phrase</div>
          <div className="phrase-text">{challenge?.phrase ?? '...'}</div>
        </div>

        <div className="field">
          <label htmlFor="enr-phrase">Verification phrase</label>
          <input
            id="enr-phrase"
            ref={bind('phrase')}
            className="mono"
            value={phraseTyped}
            onChange={(e) => setPhraseTyped(e.target.value)}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
            required
          />
          <div className="hint">Capitalisation is not checked. A new phrase is issued each round.</div>
        </div>

        <button className="primary" type="submit" disabled={busy || !challenge}>
          {busy ? 'Recording...' : 'Submit round'}
        </button>
      </form>
    </div>
  );
}

function QualityReadout({ quality }: { quality: Record<string, unknown> }) {
  const entries = Object.entries(quality);
  if (entries.length === 0) return null;
  return (
    <div className="metrics">
      {entries.map(([key, value]) => (
        <div className="metric" key={key}>
          <div className="metric-label">{key.replace(/_/g, ' ')}</div>
          <div className="metric-value">{formatValue(value)}</div>
        </div>
      ))}
    </div>
  );
}

function formatValue(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}
