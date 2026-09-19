import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import { useBehaviorCollector } from '../hooks/useBehaviorCollector';
import { ApiError, api } from '../services/api';
import type { Challenge, Decision } from '../types/api';

type Phase = 'idle' | 'analyzing' | 'result';

/**
 * The dummy login page.
 *
 * The challenge is fetched on mount, before a password is typed, so behaviour
 * is captured across the whole form fill rather than only after the credential
 * is known. That endpoint takes no password and answers identically for real
 * and unknown usernames, so it cannot be used to enumerate accounts.
 */
export default function LoginPage() {
  const { collector, bind } = useBehaviorCollector();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [phraseTyped, setPhraseTyped] = useState('');

  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);

  const usernameRef = useRef<HTMLInputElement | null>(null);
  // The challenge is bound to the username that was claimed when it was issued,
  // so we must remember which one that was.
  const claimedUsername = useRef('');

  const loadChallenge = useCallback(
    async (claim: string) => {
      setError(null);
      try {
        const next = await api.loginChallenge(claim);
        setChallenge(next);
        claimedUsername.current = claim;
        setPhraseTyped('');
        collector.reset();
      } catch (err) {
        setChallenge(null);
        setError(err instanceof ApiError ? err.message : 'Could not start a login attempt.');
      }
    },
    [collector],
  );

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!challenge || phase === 'analyzing') return;

    collector.markSubmit();
    const session = collector.snapshot(challenge.nonce, phraseTyped);

    setPhase('analyzing');
    setError(null);
    try {
      const verdict = await api.submitLogin(claimedUsername.current, password, session);
      setDecision(verdict);
      setPhase('result');
    } catch (err) {
      // A transport or server failure is an error, never an acceptance.
      setError(err instanceof ApiError ? err.message : 'The login attempt could not be completed.');
      setPhase('idle');
      await loadChallenge(claimedUsername.current || username);
    }
  }

  async function handleRetry() {
    setDecision(null);
    setPassword('');
    setPhase('idle');
    await loadChallenge(username);
    usernameRef.current?.focus();
  }

  // A challenge is single-use and expiring, so it is fetched once the user has
  // committed to a username rather than on every keystroke.
  function handleUsernameBlur() {
    const claim = username.trim().toLowerCase();
    if (claim.length >= 3 && claim !== claimedUsername.current) void loadChallenge(claim);
  }

  useEffect(() => {
    usernameRef.current?.focus();
  }, []);

  if (phase === 'result' && decision) {
    return <VerdictPanel decision={decision} onRetry={handleRetry} />;
  }

  return (
    <div className="panel">
      <h1 className="panel-title">Sign in</h1>
      <p className="panel-sub">
        Protected by behavioural authentication. A correct password alone is not
        enough to sign in.
      </p>

      {error ? <div className="notice error">{error}</div> : null}

      {phase === 'analyzing' ? (
        <div className="analyzing">
          <div className="spinner" />
          Analysing behavioural signature...
        </div>
      ) : (
        <form onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="login-username">Username</label>
            <input
              id="login-username"
              ref={(el) => {
                usernameRef.current = el;
                bind('username')(el);
              }}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onBlur={handleUsernameBlur}
              autoComplete="username"
              required
            />
          </div>

          <div className="field">
            <label htmlFor="login-password">Password</label>
            <input
              id="login-password"
              ref={bind('password')}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </div>

          <div className="phrase-card">
            <div className="phrase-label">Type this phrase</div>
            <div className="phrase-text">
              {challenge?.phrase ?? 'Enter your username to receive a phrase'}
            </div>
          </div>

          <div className="field">
            <label htmlFor="login-phrase">Verification phrase</label>
            <input
              id="login-phrase"
              ref={bind('phrase')}
              className="mono"
              value={phraseTyped}
              onChange={(e) => setPhraseTyped(e.target.value)}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              disabled={!challenge}
              required
            />
          </div>

          <button className="primary" type="submit" disabled={!challenge}>
            Sign in
          </button>
        </form>
      )}

      <p className="hint" style={{ marginTop: 18 }}>
        No account yet? <Link to="/enroll" className="muted-link">Enroll a behavioural profile</Link>.
      </p>
    </div>
  );
}

function VerdictPanel({ decision, onRetry }: { decision: Decision; onRetry: () => void }) {
  const allowed = decision.decision === 'ALLOW';

  return (
    <>
      <div className={`verdict ${allowed ? 'allow' : 'block'}`}>
        <div className="verdict-title">{allowed ? 'ACCESS GRANTED' : 'ACCESS BLOCKED'}</div>
        <div className="verdict-msg">{decision.message}</div>
        <div className="reason-code">{decision.reason}</div>
      </div>

      {decision.signals.length > 0 ? (
        <div className="panel">
          <h2 className="panel-title" style={{ fontSize: 15 }}>
            Contributing signals
          </h2>
          <div className="signal-list">
            {decision.signals.map((signal) => (
              <div className="signal" key={signal.code}>
                <div>
                  <div className="signal-label">{signal.label}</div>
                  <div className="signal-detail">{signal.detail}</div>
                </div>
                <div className={`band ${signal.band}`}>{signal.band}</div>
              </div>
            ))}
          </div>

          {/* Categories, not numbers. The exact identity and automation
              scores are withheld from whoever is attempting the login so this
              page cannot be used to tune an impersonation; they are served to
              the operator dashboard instead. */}
          <div className="metrics">
            <Metric label="Integrity" value={decision.integrity_status} />
            <Metric label="Signal coverage" value={decision.coverage_band} />
            <Metric label="Decision latency" value={`${decision.latency.total_ms.toFixed(1)} ms`} />
          </div>
        </div>
      ) : null}

      <div className="panel">
        <button className="ghost" onClick={onRetry}>
          Try another login
        </button>
      </div>
    </>
  );
}

function Metric({ label, value }: { label: string; value: number | string | null }) {
  const shown =
    value === null ? '--' : typeof value === 'number' ? value.toFixed(3) : value;
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{shown}</div>
    </div>
  );
}
