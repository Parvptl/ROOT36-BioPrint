import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError, api } from '../services/api';
import type { AttemptLog, Dashboard } from '../types/api';

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
      <div className="panel">
        <h1 className="panel-title">Operator console</h1>
        <p className="panel-sub">
          This view shows the exact per-attempt scores that the login page withholds.
          It is authenticated separately and is disabled entirely when no operator key
          is configured.
        </p>
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
              required
            />
            <div className="hint">Set as BIOPRINT_OPERATOR_KEY in backend/.env</div>
          </div>
          <button className="primary" type="submit">
            Open console
          </button>
        </form>
      </div>
    );
  }

  const latest = data?.latest ?? null;

  return (
    <>
      <div className="console-head">
        <h1 className="panel-title" style={{ margin: 0 }}>
          Security console
        </h1>
        <label className="live-toggle">
          <input
            type="checkbox"
            checked={live}
            onChange={(e) => setLive(e.target.checked)}
          />
          live
        </label>
      </div>

      {error ? <div className="notice error">{error}</div> : null}

      {latest ? <LatestVerdict attempt={latest} /> : (
        <div className="panel">
          <p className="panel-sub" style={{ margin: 0 }}>
            No authentication attempts recorded yet. Try a login.
          </p>
        </div>
      )}

      {data && data.attempts.length > 0 ? (
        <div className="panel">
          <h2 className="panel-title" style={{ fontSize: 15 }}>
            Recent attempts
          </h2>
          <div className="ledger">
            <div className="ledger-row ledger-head">
              <span>#</span>
              <span>account</span>
              <span>verdict</span>
              <span>reason</span>
              <span>identity</span>
              <span>automation</span>
              <span>latency</span>
            </div>
            {data.attempts.map((a) => (
              <div className="ledger-row" key={a.attempt_id}>
                <span className="dim">{a.attempt_id}</span>
                <span>{a.username}</span>
                <span className={a.decision === 'ALLOW' ? 'ok' : 'bad'}>{a.decision}</span>
                <span className="dim">{a.reason}</span>
                <span>{fmt(a.identity_score)}</span>
                <span>{fmt(a.automation_score)}</span>
                <span className="dim">{a.latency_ms ? `${a.latency_ms.toFixed(0)}ms` : '--'}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
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
