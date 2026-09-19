/**
 * The browser-to-server behavioural event contract.
 *
 * Mirrors backend/app/models/events.py exactly. If you change a field here,
 * change it there: the server validates this shape strictly and rejects
 * unknown keys rather than ignoring them.
 *
 * Note what is absent. There is no score, no feature vector and no decision
 * in this file. The browser is a sensor. Every number the risk engine acts on
 * is recomputed server-side from these raw events, so patching this code in
 * DevTools does not move the authentication outcome.
 */

/** Which field an event happened in. Drives per-field features and the privacy rule. */
export type FieldContext = 'username' | 'password' | 'phrase' | 'page';

/**
 * Coarse key categories.
 *
 * `char` covers every printable character. That collapse is what lets us
 * record password typing rhythm without recording the password: not even the
 * character *class* (letter vs digit vs symbol) survives.
 */
export type KeyClass =
  | 'char'
  | 'backspace'
  | 'shift_left'
  | 'shift_right'
  | 'nav'
  | 'enter'
  | 'other';

/** How focus arrived in a field. Tab-vs-pointer habit is stable and free to record. */
export type FocusVia = 'tab' | 'pointer' | 'programmatic' | 'unknown';

interface EventBase {
  /** Milliseconds since the collector started, from performance.now(). */
  t: number;
  /** event.isTrusted. One low-weight automation signal among many, never a gate. */
  trusted: boolean;
}

export interface KeyEvent extends EventBase {
  type: 'keydown' | 'keyup';
  ctx: FieldContext;
  key_class: KeyClass;
  /**
   * The physical key (KeyboardEvent.code), e.g. "KeyT".
   *
   * Populated ONLY for contexts whose text is public: the username field and
   * the server-generated challenge phrase. Always null in the password
   * context — the server rejects the payload otherwise.
   */
  code: string | null;
  repeat: boolean;
}

export interface PointerMoveEvent extends EventBase {
  type: 'pointermove';
  x: number;
  y: number;
}

export interface PointerButtonEvent extends EventBase {
  type: 'pointerdown' | 'pointerup';
  ctx: FieldContext;
  x: number;
  y: number;
  button: number;
}

export interface FocusEvent_ extends EventBase {
  type: 'focus' | 'blur';
  ctx: FieldContext;
  via: FocusVia;
}

/**
 * Fired whenever a field's value changes.
 *
 * Retained because an `input` with no preceding `keydown` is the signature of
 * `element.value = '...'`, the cheapest scripted fill there is. Only the
 * resulting length is recorded, never the content.
 */
export interface InputEvent_ extends EventBase {
  type: 'input';
  ctx: FieldContext;
  length: number;
}

export interface SimpleEvent extends EventBase {
  type: 'submit' | 'paste';
  ctx: FieldContext;
}

export type BehaviorEvent =
  | KeyEvent
  | PointerMoveEvent
  | PointerButtonEvent
  | FocusEvent_
  | InputEvent_
  | SimpleEvent;

/** Environment hints. Treated as hints only — all of these are spoofable. */
export interface ClientMeta {
  webdriver: boolean;
  pointer_type: 'mouse' | 'touch' | 'pen' | 'none';
  screen_w: number;
  screen_h: number;
  hardware_concurrency: number;
}

/** One captured interaction, bound to one single-use challenge nonce. */
export interface BehaviorSession {
  nonce: string;
  /**
   * What the user typed into the challenge field. The phrase is
   * server-generated and public, so echoing it back leaks nothing; the server
   * compares it against the phrase it issued to defeat replay.
   */
  phrase_typed: string;
  started_at_ms: number;
  events: BehaviorEvent[];
  meta: ClientMeta;
}
