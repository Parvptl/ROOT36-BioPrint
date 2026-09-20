/**
 * The four security layers, explained, for whoever just tried to log in.
 *
 * "ACCESS DENIED" with no reason is a bad security product: the legitimate
 * user cannot tell a typo from a lockout, and support cannot either.
 *
 * Everything shown here is DERIVED FROM THE REASON CODE the backend already
 * returns. Nothing is invented. In particular there is no confidence
 * percentage anywhere, because the backend deliberately does not return one:
 * the exact identity and automation scores are withheld from the login subject
 * so this page cannot be used to tune an impersonation. Bands (HIGH / MEDIUM / LOW) are what
 * the API actually provides, so bands are what is shown.
 */

export type Band = 'HIGH' | 'MEDIUM' | 'LOW';

export type Signal = {
  code: string;
  label: string;
  band: Band;
  detail: string;
};

export type Decision = {
  decision: 'ALLOW' | 'BLOCK';
  reason: string;
  message: string;
  headline?: string | null;
  integrity_status?: string | null;
  coverage_band?: string | null;
  signals: Signal[];
  latency: { total_ms: number };
  attempt_id?: number | null;
};

/** Reason codes that mean the capture could not be trusted as fresh. */
const INTEGRITY_CODES = new Set([
  'CHALLENGE_UNKNOWN',
  'CHALLENGE_EXPIRED',
  'CHALLENGE_REUSED',
  'CHALLENGE_WRONG_USER',
  'PHRASE_MISMATCH',
  'MALFORMED_EVENT_STREAM',
  'TIMESTAMP_INCONSISTENT',
]);

/** Reason codes that mean the identity comparison rejected the attempt. */
const IDENTITY_CODES = new Set([
  'BEHAVIORAL_MISMATCH',
  'KEYSTROKE_MISMATCH',
  'POINTER_MISMATCH',
  'INTERACTION_MISMATCH',
]);

type LayerState = 'pass' | 'fail' | 'unknown';

type Layer = {
  name: string;
  value: string;
  state: LayerState;
  note?: string;
};

/**
 * Which layer stopped the attempt.
 *
 * The backend evaluates gates in a fixed order — credential, integrity,
 * automation, coverage, identity — and returns the reason of the FIRST one
 * that failed. So a layer after the failing one was never evaluated, and
 * claiming it "passed" would be a lie. Those are reported as unknown.
 */
export function describeLayers(decision: Decision): Layer[] {
  const { reason } = decision;
  const allowed = decision.decision === 'ALLOW';

  const credentialFailed = reason === 'INVALID_CREDENTIALS';
  const notEnrolled = reason === 'ACCOUNT_NOT_ENROLLED';
  const integrityFailed = INTEGRITY_CODES.has(reason);
  const automationFailed = reason === 'AUTOMATION_DETECTED';
  const thinCapture = reason === 'INSUFFICIENT_SIGNAL';
  const identityFailed = IDENTITY_CODES.has(reason);

  const password: Layer = credentialFailed
    ? { name: 'Password', value: 'INVALID', state: 'fail' }
    : { name: 'Password', value: 'VALID', state: 'pass' };

  // Everything below the credential gate is only meaningful once it passed.
  if (credentialFailed) {
    return [
      password,
      { name: 'Behavioural checks', value: 'NOT REACHED', state: 'unknown',
        note: 'The credential is checked first, so no behavioural analysis ran.' },
    ];
  }

  if (notEnrolled) {
    return [
      password,
      { name: 'Behavioural profile', value: 'NOT ENROLLED', state: 'fail',
        note: 'This account has no behavioural baseline yet.' },
    ];
  }

  const freshness: Layer = integrityFailed
    ? { name: 'Replay check', value: 'FAILED', state: 'fail',
        note: 'This interaction could not be accepted as a fresh response to the challenge.' }
    : { name: 'Replay check', value: 'PASSED', state: 'pass' };

  if (integrityFailed) return [password, freshness];

  const human: Layer = automationFailed
    ? { name: 'Human presence', value: 'NOT VERIFIED', state: 'fail',
        note: 'The interaction did not resemble human motor behaviour.' }
    : { name: 'Human presence', value: 'VERIFIED', state: 'pass' };

  if (automationFailed) return [password, freshness, human];

  if (thinCapture) {
    return [
      password, freshness, human,
      { name: 'Behaviour match', value: 'NOT COMPARED', state: 'unknown',
        note: 'Too little of the profile could be compared to reach a decision.' },
    ];
  }

  // Identity was actually evaluated. The band comes from the strongest
  // contributing identity signal the backend reported — its own categories,
  // not a number we made up. A LOW deviation band is a HIGH match.
  const identityBands = decision.signals
    .filter((s) => IDENTITY_CODES.has(s.code))
    .map((s) => s.band);
  const worst: Band = identityBands.includes('HIGH')
    ? 'HIGH'
    : identityBands.includes('MEDIUM')
      ? 'MEDIUM'
      : 'LOW';

  const match: Layer = allowed
    ? {
        name: 'Behaviour match',
        value: worst === 'LOW' ? 'HIGH' : worst === 'MEDIUM' ? 'MEDIUM' : 'LOW',
        state: 'pass',
      }
    : {
        name: 'Behaviour match',
        value: 'LOW',
        state: 'fail',
        note: identityFailed
          ? 'Typing and interaction patterns did not match the enrolled profile.'
          : undefined,
      };

  return [password, freshness, human, match];
}

/** One-line statement of why, for a blocked attempt. */
export function blockSummary(decision: Decision): string | null {
  if (decision.decision === 'ALLOW') return null;
  const { reason } = decision;
  if (reason === 'INVALID_CREDENTIALS') return 'Incorrect username or password';
  if (reason === 'ACCOUNT_NOT_ENROLLED') return 'No behavioural profile for this account';
  if (reason === 'AUTOMATION_DETECTED') return 'Automated traffic detected';
  if (INTEGRITY_CODES.has(reason)) return 'Behavioural interaction cannot be reused';
  if (reason === 'INSUFFICIENT_SIGNAL') return 'Not enough interaction captured to decide';
  if (IDENTITY_CODES.has(reason)) return 'Behavioural identity mismatch';
  return decision.message;
}

/**
 * One accurate line under the verdict.
 *
 * Not the backend headline, which is usually a restatement of the title, and
 * not a fixed string: "Behavioural identity verification complete" is simply
 * untrue when the credential gate rejected the attempt before any behavioural
 * analysis ran. A security panel that overstates which checks executed is
 * worse than one that says nothing.
 */
function subtitle(decision: Decision): string {
  const { reason } = decision;
  if (decision.decision === 'ALLOW') {
    return 'All four security layers passed';
  }
  if (reason === 'INVALID_CREDENTIALS') {
    return 'Stopped at the credential check';
  }
  if (reason === 'ACCOUNT_NOT_ENROLLED') {
    return 'No behavioural baseline exists for this account yet';
  }
  if (INTEGRITY_CODES.has(reason)) {
    return 'Stopped at the replay check';
  }
  if (reason === 'AUTOMATION_DETECTED') {
    return 'Stopped at the human-presence check';
  }
  if (reason === 'INSUFFICIENT_SIGNAL') {
    return 'Too little interaction was captured to decide';
  }
  return 'Stopped at the behavioural identity check';
}

export function SecurityVerdict({ decision }: { decision: Decision }) {
  const allowed = decision.decision === 'ALLOW';
  const layers = describeLayers(decision);
  const summary = blockSummary(decision);

  return (
    <div className={`verdict ${allowed ? 'allow' : 'block'}`} style={{ marginBottom: 24 }}>
      <div className="verdict-title">
        {allowed ? 'ACCESS GRANTED' : 'ACCESS BLOCKED'}
      </div>
      <div className="verdict-msg">{subtitle(decision)}</div>

      <ul className="sec-layers">
        {layers.map((layer) => (
          <li className={`sec-layer ${layer.state}`} key={layer.name}>
            <span className="sec-layer-name">
              <span className="sec-layer-icon" aria-hidden="true">
                {layer.state === 'pass' && (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                )}
                {layer.state === 'fail' && (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                )}
                {layer.state === 'unknown' && (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                )}
              </span>
              {layer.name}
            </span>
            <span className="sec-layer-dots" aria-hidden="true" />
            <span className="sec-layer-value">{layer.value}</span>
            {layer.note ? <span className="sec-layer-note">{layer.note}</span> : null}
          </li>
        ))}
      </ul>

      {summary ? (
        <div className="sec-reason">
          <span className="sec-reason-text">{summary}</span>
          <code className="reason-code">{decision.reason}</code>
        </div>
      ) : null}
    </div>
  );
}
