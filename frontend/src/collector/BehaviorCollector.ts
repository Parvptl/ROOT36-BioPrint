/**
 * Captures the raw interaction stream for one enrollment round or login attempt.
 *
 * Design notes
 * ------------
 * Listeners are attached once at the document level in the capture phase and
 * resolve their field context by looking the event target up in a registry.
 * Per-element listeners would be re-attached on every React re-render and
 * would silently miss events fired at a field before its effect ran.
 *
 * Timestamps come from performance.now(): monotonic, unaffected by wall-clock
 * adjustment, and sub-millisecond. Date.now() would be both coarser and liable
 * to jump mid-capture.
 *
 * The collector computes nothing. It records, redacts and hands the stream to
 * the server.
 */

import { classifyKey, redactForContext } from './keyClass';
import type {
  BehaviorEvent,
  BehaviorSession,
  ClientMeta,
  FieldContext,
  FocusVia,
} from './types';

/** Hard ceiling on a capture. The server independently caps at 20 000. */
const MAX_EVENTS = 8000;
/** Pointer moves are the only high-frequency source; bounded separately so a
 *  long mouse wander cannot crowd out the rarer, higher-value key events. */
const MAX_POINTER_MOVES = 2500;
/** Minimum spacing between recorded pointer samples. ~120 Hz, below the rate
 *  at which a mouse reports, so velocity and tremor survive the thinning. */
const POINTER_MIN_INTERVAL_MS = 8;
const POINTER_MIN_DISTANCE_PX = 2;

/** A focus within this window of a Tab press is attributed to the keyboard. */
const TAB_ATTRIBUTION_MS = 150;
/** A focus within this window of a pointer press is attributed to the pointer. */
const POINTER_ATTRIBUTION_MS = 400;

export class BehaviorCollector {
  private events: BehaviorEvent[] = [];
  private fields = new Map<Element, FieldContext>();
  private running = false;

  private originMs = 0;
  private startedAtMs = 0;

  private pointerMoveCount = 0;
  private lastPointerSampleT = -Infinity;
  private lastPointerX = 0;
  private lastPointerY = 0;

  private lastTabPressT = -Infinity;
  private lastPointerDownT = -Infinity;
  private observedPointerType: ClientMeta['pointer_type'] = 'none';

  /** Register a field so its events carry the right context. */
  observeField(element: Element | null, ctx: FieldContext): void {
    if (element) this.fields.set(element, ctx);
  }

  unobserveField(element: Element | null): void {
    if (element) this.fields.delete(element);
  }

  get eventCount(): number {
    return this.events.length;
  }

  get isRunning(): boolean {
    return this.running;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.events = [];
    this.pointerMoveCount = 0;
    this.lastPointerSampleT = -Infinity;
    this.lastTabPressT = -Infinity;
    this.lastPointerDownT = -Infinity;
    this.observedPointerType = 'none';
    this.originMs = performance.now();
    this.startedAtMs = Date.now();

    // Capture phase: we see the event before any handler can stopPropagation it.
    document.addEventListener('keydown', this.onKeyDown, true);
    document.addEventListener('keyup', this.onKeyUp, true);
    document.addEventListener('pointermove', this.onPointerMove, true);
    document.addEventListener('pointerdown', this.onPointerDown, true);
    document.addEventListener('pointerup', this.onPointerUp, true);
    document.addEventListener('focusin', this.onFocusIn, true);
    document.addEventListener('focusout', this.onFocusOut, true);
    document.addEventListener('input', this.onInput, true);
    document.addEventListener('paste', this.onPaste, true);
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    document.removeEventListener('keydown', this.onKeyDown, true);
    document.removeEventListener('keyup', this.onKeyUp, true);
    document.removeEventListener('pointermove', this.onPointerMove, true);
    document.removeEventListener('pointerdown', this.onPointerDown, true);
    document.removeEventListener('pointerup', this.onPointerUp, true);
    document.removeEventListener('focusin', this.onFocusIn, true);
    document.removeEventListener('focusout', this.onFocusOut, true);
    document.removeEventListener('input', this.onInput, true);
    document.removeEventListener('paste', this.onPaste, true);
  }

  /** Discard everything captured so far and restart the clock. */
  reset(): void {
    const wasRunning = this.running;
    this.stop();
    this.events = [];
    if (wasRunning) this.start();
  }

  /**
   * Record that the user submitted, then package the capture.
   *
   * `phraseTyped` is the literal contents of the challenge field. That text is
   * server-generated and public, so returning it leaks nothing, and the server
   * needs it to confirm the capture answers the challenge it actually issued.
   */
  snapshot(nonce: string, phraseTyped: string): BehaviorSession {
    return {
      nonce,
      phrase_typed: phraseTyped,
      started_at_ms: this.startedAtMs,
      events: [...this.events],
      meta: this.buildMeta(),
    };
  }

  markSubmit(ctx: FieldContext = 'page'): void {
    this.push({ type: 'submit', t: this.now(), trusted: true, ctx });
  }

  // ---------------------------------------------------------------- internals

  private now(): number {
    return Math.max(0, performance.now() - this.originMs);
  }

  private push(event: BehaviorEvent): void {
    if (this.events.length >= MAX_EVENTS) return;
    this.events.push(event);
  }

  private contextFor(target: EventTarget | null): FieldContext {
    if (target instanceof Element) {
      const ctx = this.fields.get(target);
      if (ctx) return ctx;
    }
    return 'page';
  }

  private buildMeta(): ClientMeta {
    const nav = navigator as Navigator & { hardwareConcurrency?: number };
    return {
      // Recorded for the audit trail. Trivially removed by any real attacker,
      // so the detector weights it near zero rather than gating on it.
      webdriver: Boolean(nav.webdriver),
      pointer_type: this.observedPointerType,
      screen_w: window.screen?.width ?? 0,
      screen_h: window.screen?.height ?? 0,
      hardware_concurrency: nav.hardwareConcurrency ?? 0,
    };
  }

  private onKeyDown = (event: Event): void => {
    const e = event as KeyboardEvent;
    const t = this.now();
    if (e.code === 'Tab') this.lastTabPressT = t;

    const ctx = this.contextFor(e.target);
    const { code, key_class } = redactForContext(ctx, e.code, classifyKey(e));

    this.push({
      type: 'keydown',
      t,
      trusted: e.isTrusted,
      ctx,
      key_class,
      code,
      // Auto-repeat from a held key is not a deliberate press; timing features
      // exclude these rather than treating them as impossibly fast typing.
      repeat: e.repeat,
    });
  };

  private onKeyUp = (event: Event): void => {
    const e = event as KeyboardEvent;
    const ctx = this.contextFor(e.target);
    const { code, key_class } = redactForContext(ctx, e.code, classifyKey(e));

    this.push({
      type: 'keyup',
      t: this.now(),
      trusted: e.isTrusted,
      ctx,
      key_class,
      code,
      repeat: false,
    });
  };

  private onPointerMove = (event: Event): void => {
    const e = event as PointerEvent;
    if (this.pointerMoveCount >= MAX_POINTER_MOVES) return;

    const t = this.now();
    const dx = e.clientX - this.lastPointerX;
    const dy = e.clientY - this.lastPointerY;
    const movedEnough = Math.abs(dx) >= POINTER_MIN_DISTANCE_PX || Math.abs(dy) >= POINTER_MIN_DISTANCE_PX;
    if (t - this.lastPointerSampleT < POINTER_MIN_INTERVAL_MS && !movedEnough) return;

    this.lastPointerSampleT = t;
    this.lastPointerX = e.clientX;
    this.lastPointerY = e.clientY;
    this.pointerMoveCount += 1;
    this.notePointerType(e);

    this.push({
      type: 'pointermove',
      t,
      trusted: e.isTrusted,
      x: Math.round(e.clientX),
      y: Math.round(e.clientY),
    });
  };

  private onPointerDown = (event: Event): void => {
    const e = event as PointerEvent;
    const t = this.now();
    this.lastPointerDownT = t;
    this.notePointerType(e);

    this.push({
      type: 'pointerdown',
      t,
      trusted: e.isTrusted,
      ctx: this.contextFor(e.target),
      x: Math.round(e.clientX),
      y: Math.round(e.clientY),
      button: Math.max(0, Math.min(4, e.button)),
    });
  };

  private onPointerUp = (event: Event): void => {
    const e = event as PointerEvent;
    this.push({
      type: 'pointerup',
      t: this.now(),
      trusted: e.isTrusted,
      ctx: this.contextFor(e.target),
      x: Math.round(e.clientX),
      y: Math.round(e.clientY),
      button: Math.max(0, Math.min(4, e.button)),
    });
  };

  private onFocusIn = (event: Event): void => {
    const t = this.now();
    this.push({
      type: 'focus',
      t,
      trusted: event.isTrusted,
      ctx: this.contextFor(event.target),
      via: this.attributeFocus(t),
    });
  };

  private onFocusOut = (event: Event): void => {
    this.push({
      type: 'blur',
      t: this.now(),
      trusted: event.isTrusted,
      ctx: this.contextFor(event.target),
      via: 'unknown',
    });
  };

  private onInput = (event: Event): void => {
    const target = event.target as HTMLInputElement | null;
    this.push({
      type: 'input',
      t: this.now(),
      trusted: event.isTrusted,
      ctx: this.contextFor(event.target),
      // Length only. The value itself is never read into the event stream —
      // not for the password field, not for any field.
      length: target?.value?.length ?? 0,
    });
  };

  private onPaste = (event: Event): void => {
    this.push({
      type: 'paste',
      t: this.now(),
      trusted: event.isTrusted,
      ctx: this.contextFor(event.target),
    });
  };

  /**
   * Attribute a focus change to the keyboard, the pointer, or neither.
   *
   * Whether someone Tabs between fields or clicks each one is a habit that
   * barely varies within a person and varies a lot between people, and it
   * costs nothing to observe. "programmatic" covers focus moved by script,
   * including our own autofocus.
   */
  private attributeFocus(t: number): FocusVia {
    if (t - this.lastTabPressT <= TAB_ATTRIBUTION_MS) return 'tab';
    if (t - this.lastPointerDownT <= POINTER_ATTRIBUTION_MS) return 'pointer';
    return 'programmatic';
  }

  private notePointerType(e: PointerEvent): void {
    const kind = e.pointerType;
    if (kind === 'mouse' || kind === 'touch' || kind === 'pen') {
      this.observedPointerType = kind;
    }
  }
}
