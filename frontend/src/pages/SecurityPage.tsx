import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError, api } from '../services/api';
import type { AttemptLog, Dashboard } from '../types/api';
import PageFrame from '../components/PageFrame';

const POLL_MS = 2000;
const KEY_STORAGE = 'bioprint.operatorKey';

/**
 * Operator console.
 *
 * This is the only place the exact identity and automation scores are shown.
 * The login page deliberately shows categories instead, so that whoever is
 * attempting a login cannot read their distance from acceptance and tune
 * toward it. The operator, who is not the subject of the decision, can.
 *
 * The key is held in sessionStorage so a demo survives a page reload without
 * leaving it on disk afterwards.
 */
export default function SecurityPage() {
  const [operatorKey, setOperatorKey] = useState(
    () => sessionStorage.getItem(KEY_STORAGE) ?? '',
  );
  const [keyInput, setKeyInput] = useState('');
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState(true);
  const [query, setQuery] = useState('');
  const [decisionFilter, setDecisionFilter] = useState<'ALL' | 'ALLOW' | 'BLOCK'>('ALL');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(
    async (key: string) => {
      try {
        setData(await api.dashboard(key));
        setError(null);
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.status === 404
              ? 'The dashboard is disabled. Set BIOPRINT_OPERATOR_KEY in backend/.env and restart.'
              : err.status === 401
                ? 'That operator key was not accepted.'
                : err.message
            : 'Could not reach the BioPrint service.';
        setError(message);
        if (err instanceof ApiError && (err.status === 401 || err.status === 404)) {
          sessionStorage.removeItem(KEY_STORAGE);
          setOperatorKey('');
        }
      }
    },
    [],
  );

  useEffect(() => {
    if (!operatorKey) return;
    void refresh(operatorKey);
    if (!live) return;

    timer.current = window.setInterval(() => void refresh(operatorKey), POLL_MS);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, [operatorKey, live, refresh]);

  function unlock(event: React.FormEvent) {
    event.preventDefault();
    sessionStorage.setItem(KEY_STORAGE, keyInput);
    setOperatorKey(keyInput);
    setKeyInput('');
  }

  if (!operatorKey) {
    return (
      <PageFrame eyebrow="Restricted operator workspace" title="Open security intelligence" description="Exact decision telemetry is available only to authorized operators, never to a login subject.">
      <div className="panel">
        <div className="gate-icon" aria-hidden="true">
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" opacity=".2"/>
            <rect x="8" y="11" width="8" height="6" rx="1" />
            <path d="M10 11V9a2 2 0 0 1 4 0v2" />
          </svg>
        </div>
        {error ? <div className="notice error">{error}</div> : null}
        <form onSubmit={unlock}>
          <div className="field">
            <label htmlFor="op-key">Operator key</label>
            <input
              id="op-key"
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              autoComplete="off"
              placeholder="Enter your operator access key"
              required
            />
            <div className="hint">Set as BIOPRINT_OPERATOR_KEY in backend/.env</div>
          </div>
          <button className="primary" type="submit">
            Open console
          </button>
        </form>
      </div>
      </PageFrame>
    );
  }

  const latest = data?.latest ?? null;
  const attempts = data?.attempts ?? [];
  const blocked = attempts.filter((attempt) => attempt.decision === 'BLOCK').length;
  const allowed = attempts.filter((attempt) => attempt.decision === 'ALLOW').length;
  const visibleAttempts = attempts.filter((attempt) =>
    (decisionFilter === 'ALL' || attempt.decision === decisionFilter) &&
    `${attempt.username} ${attempt.reason} ${attempt.attempt_id}`.toLowerCase().includes(query.toLowerCase()),
  );
  const selected = attempts.find((attempt) => attempt.attempt_id === selectedId) ?? latest;

  return (
    <PageFrame eyebrow="Restricted operator workspace" title="Security intelligence" description="Investigate retained authentication telemetry without exposing decision thresholds to the login subject.">
      <div className="console-head">
        <div>
          <div className="console-kicker">Operator workspace</div>
          <h1 className="panel-title" style={{ margin: 0 }}>Security intelligence</h1>
        </div>
        <label className="live-toggle">
          <input
            type="checkbox"
            checked={live}
            onChange={(e) => setLive(e.target.checked)}
          />
          <span className={`live-indicator ${live ? 'active' : ''}`} />
          {live ? 'live' : 'paused'}
        </label>
      </div>

      {error ? <div className="notice error">{error}</div> : null}

      <section className="console-summary" aria-label="Security overview">
        <div className="console-hero">
          <div className="console-kicker">Behavioural access control</div>
          <h2>Identity signals, evaluated in context.</h2>
          <p>Every authentication attempt is checked for behavioural similarity, automation, and session integrity.</p>
        </div>
        <Summary label="Observed attempts" value={String(attempts.length)} note="In retained audit history" />
        <Summary label="Blocked attempts" value={String(blocked)} note={blocked ? 'Review latest investigation' : 'No blocked attempts recorded'} tone={blocked ? 'alert' : undefined} />
        <Summary label="Access granted" value={String(allowed)} note="Behavioural verification passed" tone="good" />
      </section>

      {!data && !error ? <ConsoleSkeleton /> : null}
      {latest ? <LatestVerdict attempt={latest} /> : data ? (
        <div className="panel">
          <div className="empty-state">
            <strong>No authentication attempts recorded</strong>
            <span>Try a login from the Secure Access page to see behavioural evaluation results here.</span>
          </div>
        </div>
      ) : null}

      {selected ? <InvestigationPanel attempt={selected} onClose={() => setSelectedId(null)} /> : null}

      {data && data.attempts.length > 0 ? (
        <div className="panel">
          <div className="table-heading">
            <div>
              <h2 className="panel-title" style={{ fontSize: 15 }}>Authentication audit trail</h2>
              <p>Choose a record to inspect the real decision signals.</p>
            </div>
            <span>{visibleAttempts.length} shown</span>
          </div>
          <div className="audit-controls">
            <input aria-label="Search authentication audit trail" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search identity or reason" />
            <div className="filter-group" aria-label="Filter by decision">
              {(['ALL', 'ALLOW', 'BLOCK'] as const).map((filter) => <button key={filter} className={decisionFilter === filter ? 'selected' : ''} onClick={() => setDecisionFilter(filter)}>{filter === 'ALL' ? 'All' : filter === 'ALLOW' ? 'Granted' : 'Blocked'}</button>)}
            </div>
          </div>
          <div className="ledger">
            <div className="ledger-row ledger-head">
              <span>#</span>
              <span>account</span>
              <span>verdict</span>
              <span>reason</span>
              <span>identity</span>
              <span>statistical</span>
              <span>automation</span>
              <span>latency</span>
            </div>
            {visibleAttempts.map((a) => (
              <button className={`ledger-row audit-row ${selected?.attempt_id === a.attempt_id ? 'selected' : ''}`} key={a.attempt_id} onClick={() => setSelectedId(a.attempt_id)} aria-label={`Inspect attempt ${a.attempt_id} for ${a.username}`}>
                <span className="dim">{a.attempt_id}</span>
                <span>{a.username}</span>
                <span className={a.decision === 'ALLOW' ? 'ok' : 'bad'}>{a.decision}</span>
                <span className="dim">{a.reason}</span>
                <span>{fmt(a.identity_score)}</span>
                <span>{fmt(a.statistical_identity_score)}</span>
                <span>{fmt(a.automation_score)}</span>
                <span className="dim">{a.latency_ms ? `${a.latency_ms.toFixed(0)}ms` : '--'}</span>
              </button>
            ))}
          </div>
          {visibleAttempts.length === 0 ? <div className="empty-state"><strong>No matching audit events</strong><span>Try a different identity, reason, or decision filter.</span></div> : null}
        </div>
      ) : null}
    </PageFrame>
  );
}

function ConsoleSkeleton() {
  return <div className="console-skeleton" aria-label="Loading security intelligence" aria-busy="true">
    <div /><div /><div /><div />
  </div>;
}

function InvestigationPanel({ attempt, onClose }: { attempt: AttemptLog; onClose: () => void }) {
  const isBlocked = attempt.decision === 'BLOCK';

  // Build explainability factors from the actual scores
  const factors: { label: string; value: string; severity: 'low' | 'medium' | 'high'; note: string }[] = [];

  if (attempt.identity_score !== null) {
    const sev = attempt.identity_score > 0.7 ? 'high' : attempt.identity_score > 0.4 ? 'medium' : 'low';
    factors.push({ label: 'Identity deviation', value: fmt(attempt.identity_score), severity: sev, note: 'Behavioural distance from enrolled baseline' });
  }
  if (attempt.automation_score !== null) {
    const sev = attempt.automation_score > 0.7 ? 'high' : attempt.automation_score > 0.3 ? 'medium' : 'low';
    factors.push({ label: 'Automation signal', value: fmt(attempt.automation_score), severity: sev, note: 'Bot / scripted input indicators' });
  }

  return <section className={`investigation ${isBlocked ? 'blocked' : 'allowed'}`} aria-live="polite">
    <div className="investigation-head">
      <div>
        <div className="console-kicker">Selected authentication event</div>
        <h2>{attempt.username} <span>· attempt #{attempt.attempt_id}</span></h2>
      </div>
      <button className="close-investigation" onClick={onClose} aria-label="Close investigation">×</button>
    </div>

    {/* Decision banner */}
    <div className="investigation-decision">
      <div className={`decision-pill ${isBlocked ? 'block' : 'allow'}`}>{attempt.decision === 'ALLOW' ? 'Access granted' : 'Access blocked'}</div>
      <p className="investigation-reason">{attempt.reason.replace(/_/g, ' ')}</p>
    </div>

    {/* Explainability: WHY THIS WAS FLAGGED */}
    {factors.length > 0 && (
      <div className="investigation-factors">
        <div className="factor-section-label">Contributing signals</div>
        {factors.map((f) => (
          <div className="investigation-factor" key={f.label}>
            <div className="investigation-factor-info">
              <div className="factor-label">{f.label}</div>
              <div className="factor-note">{f.note}</div>
            </div>
            <div className="investigation-factor-score">
              <div className="factor-value">{f.value}</div>
              <div className={`factor-severity ${f.severity}`}>{f.severity.toUpperCase()}</div>
            </div>
          </div>
        ))}
      </div>
    )}

    <div className="investigation-grid">
      <div>
        <div className="factor-label">Identity deviation</div>
        <div className="factor-value">{fmt(attempt.identity_score)}</div>
        <div className="factor-note">Exact score is operator-only.</div>
      </div>
      <div>
        <div className="factor-label">Automation signal</div>
        <div className="factor-value">{fmt(attempt.automation_score)}</div>
        <div className="factor-note">Captured for this attempt.</div>
      </div>
      <div>
        <div className="factor-label">Decision latency</div>
        <div className="factor-value">{attempt.latency_ms ? `${attempt.latency_ms.toFixed(0)} ms` : '--'}</div>
        <div className="factor-note">End-to-end evaluation time.</div>
      </div>
      <div>
        <div className="factor-label">Integrity</div>
        <div className="factor-value">{integrityOf(attempt)}</div>
        <div className="factor-note">Challenge validation.</div>
      </div>
    </div>
  </section>;
}

function Summary({ label, value, note, tone }: { label: string; value: string; note: string; tone?: 'alert' | 'good' }) {
  return <div className="kpi">
    <div className="kpi-label">{label}</div>
    <div className="kpi-value">{value}</div>
    <div className={`kpi-note ${tone ?? ''}`}>{note}</div>
  </div>;
}

function LatestVerdict({ attempt }: { attempt: AttemptLog }) {
  const allowed = attempt.decision === 'ALLOW';
  return (
    <div className={`verdict ${allowed ? 'allow' : 'block'}`}>
      <div className="verdict-title">{headlineFor(attempt)}</div>
      <div className="verdict-msg">
        account <strong>{attempt.username}</strong> &middot; attempt #{attempt.attempt_id}
      </div>
      <div className="metrics" style={{ marginTop: 20 }}>
        <Metric label="Identity deviation" value={fmt(attempt.identity_score)} />
        <Metric label="Statistical" value={fmt(attempt.statistical_identity_score)} />
        <Metric label="Automation" value={fmt(attempt.automation_score)} />
        <Metric label="Integrity" value={integrityOf(attempt)} />
        <Metric label="Coverage" value={fmt(attempt.coverage)} />
        <Metric
          label="Latency"
          value={attempt.latency_ms ? `${attempt.latency_ms.toFixed(0)} ms` : '--'}
        />
      </div>
      <div className="reason-code">{attempt.reason}</div>
    </div>
  );
}

const INTEGRITY_REASONS = new Set([
  'CHALLENGE_REUSED', 'CHALLENGE_EXPIRED', 'CHALLENGE_UNKNOWN',
  'CHALLENGE_WRONG_USER', 'PHRASE_MISMATCH', 'TIMESTAMP_INCONSISTENT',
  'MALFORMED_EVENT_STREAM',
]);

function headlineFor(a: AttemptLog): string {
  if (a.decision === 'ALLOW') return 'ACCESS GRANTED';
  if (a.reason === 'AUTOMATION_DETECTED') return 'AUTOMATION DETECTED';
  if (INTEGRITY_REASONS.has(a.reason)) return 'REPLAY / INTEGRITY FAILURE';
  if (a.reason === 'INVALID_CREDENTIALS') return 'CREDENTIAL REJECTED';
  if (a.reason === 'INSUFFICIENT_SIGNAL') return 'INSUFFICIENT SIGNAL';
  return 'BEHAVIORAL MISMATCH';
}

function integrityOf(a: AttemptLog): string {
  if (INTEGRITY_REASONS.has(a.reason)) return 'FAIL';
  return (a.integrity_score ?? 0) >= 0.5 ? 'FAIL' : 'PASS';
}

function fmt(value: number | null): string {
  return value === null || value === undefined ? '--' : value.toFixed(3);
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
