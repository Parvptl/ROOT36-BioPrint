import { useCallback, useEffect, useRef, useState } from 'react';
import { SecurityVerdict } from '../components/SecurityVerdict';
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
  const [showAccess, setShowAccess] = useState(false);

  const usernameRef = useRef<HTMLInputElement | null>(null);
  // The challenge is bound to the username that was claimed when it was issued,
  // so we must remember which one that was.
  const claimedUsername = useRef('');

  const loadChallenge = useCallback(
    // `keepMessage` preserves an explanation the caller just set. A rejected
    // capture has to issue a fresh challenge (the old nonce is spent), and
    // clearing the error here made the failure invisible: the form simply
    // reset itself with new text and no reason, which reads as the app
    // losing your input rather than as a security decision.
    async (claim: string, keepMessage = false) => {
      if (!keepMessage) setError(null);
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
      setError(
        err instanceof ApiError
          ? `${err.message} Your interaction was not accepted, so a fresh phrase has been issued.`
          : 'The login attempt could not be completed.',
      );
      setPhase('idle');
      await loadChallenge(claimedUsername.current || username, true);
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

  if (!showAccess) {
    return <CinematicHero onVerify={() => setShowAccess(true)} />;
  }

  return (
    <div className="access-layout">
      <section className="access-intro">
        <div className="console-kicker">Adaptive identity verification</div>
        <h1>Trust the person,<br /><em>not just the password.</em></h1>
        <p>BioPrint evaluates how a login is performed—keystroke rhythm, pointer movement, and interaction flow—to protect access without adding friction.</p>
        <div className="trust-points">
          <span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{marginRight: '6px'}}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            Behavioural signals
          </span>
          <span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{marginRight: '6px'}}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            Automation defence
          </span>
          <span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{marginRight: '6px'}}><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
            Privacy-preserving
          </span>
        </div>
      </section>
      <div className="panel access-panel">
      <button className="back-link" type="button" onClick={() => setShowAccess(false)}>← Intelligence overview</button>
      <h2 className="panel-title">Verify your identity</h2>
      <p className="panel-sub">Your interaction signature adds an invisible layer of account protection.</p>

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
    </div>
  );
}

function CinematicHero({ onVerify }: { onVerify: () => void }) {
  return <section className="cinematic-hero" aria-labelledby="hero-title">
    <div className="hero-grid" aria-hidden="true" />
    <svg className="threat-topology" viewBox="0 0 800 430" fill="none" aria-hidden="true">
      <g className="topology-lines"><path d="M84 120L260 205 410 112 607 188 746 84M260 205l40 140 177-26 130-131M410 112l67 207M477 319l182 51" /><path d="M84 120l216 225M607 188l52 182" /></g>
      <g className="topology-nodes"><circle cx="84" cy="120" r="7"/><circle cx="260" cy="205" r="8"/><circle cx="410" cy="112" r="7"/><circle cx="607" cy="188" r="9"/><circle cx="746" cy="84" r="6"/><circle cx="300" cy="345" r="7"/><circle cx="477" cy="319" r="7"/><circle cx="659" cy="370" r="6"/></g>
      <circle className="threat-node" cx="607" cy="188" r="15" />
    </svg>
    <header className="hero-nav"><Link className="hero-brand" to="/login"><span>⌁</span>BioPrint <i>Security Intelligence</i></Link><nav><Link to="/enroll">Enroll identity</Link><Link to="/security">Operator console</Link></nav></header>
    <div className="hero-copy-cinematic">
      <div className="security-badge"><span className="status-dot" />Behavioural defense active</div>
      <h1 id="hero-title"><span>Know the person.</span><span>Stop <em>the intrusion.</em></span></h1>
      <p>BioPrint verifies the behavioural signature behind every authentication attempt, helping security teams identify impersonation and automation before access is granted.</p>
      <div className="hero-actions-cinematic"><button className="metal-button metal-button--solid" onClick={onVerify}>Verify identity <span>→</span></button><Link className="metal-button metal-button--ghost" to="/security">Open security console</Link></div>
    </div>
    <footer className="hero-capabilities"><div><b>01</b><span>Behavioural<br />authentication</span></div><div><b>02</b><span>Automation<br />detection</span></div><div><b>03</b><span>Operator audit<br />intelligence</span></div></footer>
  </section>;
}

/**
 * The part of the decision this system is responsible for.
 *
 * Excludes `credential_ms` — Argon2id at 64 MiB is intentionally expensive,
 * it is what makes a leaked user table costly to crack, and it dominates the
 * total. Including it in a "latency" headline measures the password hash, not
 * the behavioural engine, and it swings with whatever else the machine is
 * doing.
 */
function behaviouralMs(l: Decision['latency']): number {
  return (
    (l.validation_ms ?? 0) +
    (l.extraction_ms ?? 0) +
    (l.identity_ms ?? 0) +
    (l.automation_ms ?? 0)
  );
}

function VerdictPanel({ decision, onRetry }: { decision: Decision; onRetry: () => void }) {
  return (
    <>
      <SecurityVerdict decision={decision} />

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
            {/* Split deliberately. "Decision latency" used to show total_ms,
                which is ~100% Argon2id password hashing — a cost the design
                chooses on purpose, and one that varies more than fivefold with
                CPU contention on the host. Quoting it as the system's latency
                made a busy laptop look like a slow authentication engine.
                Behavioural analysis is the part this project actually
                controls, so it is the headline; the total is shown beside it
                and says what it includes. */}
            <Metric
              label="Behavioural analysis"
              value={`${behaviouralMs(decision.latency).toFixed(1)} ms`}
            />
            <Metric
              label="Total (incl. Argon2id)"
              value={`${decision.latency.total_ms.toFixed(1)} ms`}
            />
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
