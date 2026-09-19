/**
 * Key categorisation.
 *
 * Two jobs:
 *  - produce the coarse `key_class` the password context is limited to;
 *  - decide whether the precise `code` may be transmitted at all.
 *
 * Everything here keys off KeyboardEvent.code (the physical key) rather than
 * KeyboardEvent.key (the produced character). `code` is layout-independent, so
 * "the key under the left index finger" means the same thing regardless of the
 * user's keyboard layout, and it does not change when Shift is held.
 */

import type { FieldContext, KeyClass } from './types';

const NAV_CODES = new Set([
  'Tab',
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'ArrowDown',
  'Home',
  'End',
  'PageUp',
  'PageDown',
  'Escape',
]);

/**
 * Keys reached with the left hand on a standard QWERTY layout.
 *
 * Used for the left/right dwell asymmetry feature. Hand assignment is an
 * approximation — a hunt-and-peck typist does not use the touch-typing
 * fingering this assumes — but it is a consistent approximation *per user*,
 * which is all a behavioural comparison needs.
 */
const LEFT_HAND_CODES = new Set([
  'Backquote', 'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5',
  'KeyQ', 'KeyW', 'KeyE', 'KeyR', 'KeyT',
  'KeyA', 'KeyS', 'KeyD', 'KeyF', 'KeyG',
  'KeyZ', 'KeyX', 'KeyC', 'KeyV', 'KeyB',
  'Tab', 'CapsLock', 'ShiftLeft', 'ControlLeft', 'AltLeft',
]);

export function isLeftHand(code: string): boolean {
  return LEFT_HAND_CODES.has(code);
}

/**
 * Map a key event onto its coarse class.
 *
 * Delete is folded into `backspace`: both are corrections, and what the
 * correction features measure is error-recovery behaviour, not which key
 * performed it.
 */
export function classifyKey(event: KeyboardEvent): KeyClass {
  const code = event.code;

  if (code === 'Backspace' || code === 'Delete') return 'backspace';
  if (code === 'ShiftLeft') return 'shift_left';
  if (code === 'ShiftRight') return 'shift_right';
  if (code === 'Enter' || code === 'NumpadEnter') return 'enter';
  if (NAV_CODES.has(code)) return 'nav';

  // A single-character `key` means the press produced text.
  if (event.key.length === 1) return 'char';

  return 'other';
}

/**
 * Decide what may be transmitted for this key press.
 *
 * In the password context the precise key is dropped and Shift collapses to
 * `other`. Dropping the code is obvious. Collapsing Shift is the subtler half:
 * `shift_left` at position 4 of a password would tell an observer that
 * character 4 was capitalised, which is real structural information about the
 * secret. Shift-hand preference is instead sourced from the challenge phrase,
 * whose capitalisation is server-generated and public.
 */
export function redactForContext(
  ctx: FieldContext,
  code: string,
  keyClass: KeyClass,
): { code: string | null; key_class: KeyClass } {
  if (ctx !== 'password') {
    return { code, key_class: keyClass };
  }

  const isShift = keyClass === 'shift_left' || keyClass === 'shift_right';
  return { code: null, key_class: isShift ? 'other' : keyClass };
}
