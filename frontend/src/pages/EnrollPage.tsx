import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import PageFrame from '../components/PageFrame';
import BehaviorCaptureStatus from '../components/BehaviorCaptureStatus';

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

  const roundsTotal = progress?.sessions_required ?? challenge?.rounds_total ?? 2;
  const roundsDone = progress?.sessions_captured ?? 0;

  const requestRound = useCallback(
    // `keepMessage` preserves an explanation of why the previous round was
    // rejected. Without it the message is written and then wiped by the very
    // next request, and the user is sent back to an identical form with no
    // idea what went wrong.
    async (user: string, pass: string, keepMessage = false) => {
      setBusy(true);
      if (!keepMessage) setError(null);
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
        // Round rejected (bad phrase, automation, too little signal). Re-issue
        // rather than folding a poor-quality capture into the baseline, and
        // keep the explanation on screen through the re-issue.
        setError(result.message);
        await requestRound(username, password, true);
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
      <PageFrame eyebrow="Identity enrollment complete" title="Behavioural profile created" description="Your baseline is ready for protected authentication.">
      <div className="panel" style={{ textAlign: 'center' }}>
        <div className="enroll-done-icon" aria-hidden="true">
          <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
            <circle cx="28" cy="28" r="27" stroke="var(--allow)" strokeWidth="1.5" strokeDasharray="4 3" opacity=".4" />
            <circle cx="28" cy="28" r="20" stroke="var(--allow)" strokeWidth="2" opacity=".8" />
            <path d="M20 28l6 6 10-12" stroke="var(--allow)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          </svg>
        </div>
        <p className="panel-sub" style={{ textAlign: 'center', maxWidth: '420px', margin: '16px auto 20px' }}>
          BioPrint has built a baseline from {roundsDone} enrollment rounds for{' '}
          <strong>{username}</strong>. Future logins are compared against this profile.
        </p>
        {progress?.quality ? <QualityReadout quality={progress.quality} /> : null}
        <Link to="/login" className="ghost" style={{ display: 'inline-flex', marginTop: '8px' }}>
          Continue to login →
        </Link>
      </div>
      </PageFrame>
    );
  }

  if (stage === 'register') {
    return (
      <PageFrame eyebrow="Identity enrollment · step 1" title="Create your protected identity" description="Set up an account, then provide a short behavioural baseline for future access verification.">
      <div className="enroll-context">
        <div className="enroll-context-item">
          <div className="enroll-context-num">01</div>
          <div><strong>Register</strong><span>Create credentials</span></div>
        </div>
        <div className="enroll-context-divider" />
        <div className="enroll-context-item dim">
          <div className="enroll-context-num">02</div>
          <div><strong>Capture</strong><span>Type enrollment phrases</span></div>
        </div>
        <div className="enroll-context-divider" />
        <div className="enroll-context-item dim">
          <div className="enroll-context-num">03</div>
          <div><strong>Protected</strong><span>Profile is active</span></div>
        </div>
      </div>
      <div className="panel">

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
            {busy ? 'Creating account…' : 'Create account and begin'}
          </button>
        </form>
      </div>
      </PageFrame>
    );
  }

  // Rounds stage
  const pct = roundsTotal > 0 ? Math.round((roundsDone / roundsTotal) * 100) : 0;

  return (
    <PageFrame eyebrow="Behavioural baseline in progress" title={`Enrollment round ${Math.min(roundsDone + 1, roundsTotal)} of ${roundsTotal}`} description="Use the form naturally. BioPrint measures interaction patterns, not the content of your password.">
    <div className="enroll-context">
      <div className="enroll-context-item done">
        <div className="enroll-context-num">01</div>
        <div><strong>Register</strong><span>Complete</span></div>
      </div>
      <div className="enroll-context-divider active" />
      <div className="enroll-context-item active">
        <div className="enroll-context-num">02</div>
        <div><strong>Capture</strong><span>{roundsDone}/{roundsTotal} rounds</span></div>
      </div>
      <div className="enroll-context-divider" />
      <div className="enroll-context-item dim">
        <div className="enroll-context-num">03</div>
        <div><strong>Protected</strong><span>Awaiting baseline</span></div>
      </div>
    </div>
    <div className="panel">

      <div className="enroll-progress-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`Enrollment ${pct}% complete`}>
        <div className="enroll-progress-fill" style={{ width: `${pct}%` }} />
      </div>

      <div className="rounds">
        {Array.from({ length: roundsTotal }, (_, i) => (
          <div
            key={i}
            className={`round-pip ${i < roundsDone ? 'done' : i === roundsDone ? 'active' : ''}`}
            title={i < roundsDone ? `Round ${i+1}: captured` : i === roundsDone ? `Round ${i+1}: current` : `Round ${i+1}: pending`}
          />
        ))}
      </div>

      <BehaviorCaptureStatus collector={collector} />

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
          <div className="phrase-text">{challenge?.phrase ?? '…'}</div>
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
          {busy ? 'Recording…' : 'Submit round'}
        </button>
      </form>
    </div>
    </PageFrame>
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
