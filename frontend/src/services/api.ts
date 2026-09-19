/**
 * Thin client for the BioPrint API.
 *
 * Every call here sends raw material (credentials, event streams) and reads
 * back a verdict. Nothing in this file computes or influences a score; if it
 * did, an attacker with DevTools would own the authentication decision.
 */

import type { BehaviorSession } from '../collector';
import type {
  Challenge,
  Decision,
  EnrollmentProgress,
  ProfileStatus,
  RegisterResult,
} from '../types/api';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    // A dead backend must never be mistaken for a successful login. Callers
    // surface this as an error state, never as ACCESS GRANTED.
    throw new ApiError(0, 'Cannot reach the BioPrint service. Is the backend running?', null);
  }
  return handle<T>(response);
}

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`);
  } catch {
    throw new ApiError(0, 'Cannot reach the BioPrint service. Is the backend running?', null);
  }
  return handle<T>(response);
}

async function handle<T>(response: Response): Promise<T> {
  const text = await response.text();
  let payload: unknown = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text;
  }

  if (!response.ok) {
    const detail =
      typeof payload === 'object' && payload !== null && 'detail' in payload
        ? (payload as { detail: unknown }).detail
        : payload;
    throw new ApiError(response.status, describe(detail, response.status), detail);
  }
  return payload as T;
}

function describe(detail: unknown, status: number): string {
  if (typeof detail === 'string') return detail;
  if (status === 429) return 'Too many attempts. Please wait before trying again.';
  if (status === 422) return 'The request was rejected as malformed.';
  return `Request failed (${status}).`;
}

export const api = {
  register: (username: string, password: string, consent: boolean) =>
    post<RegisterResult>('/auth/register', { username, password, consent }),

  enrollmentChallenge: (username: string, password: string) =>
    post<Challenge>('/auth/enrollment/start', { username, password }),

  submitEnrollmentRound: (username: string, session: BehaviorSession) =>
    post<EnrollmentProgress>('/auth/enrollment/submit', { username, session }),

  loginChallenge: (username: string) =>
    post<Challenge>('/auth/login/challenge', { username }),

  submitLogin: (username: string, password: string, session: BehaviorSession) =>
    post<Decision>('/auth/login/behavior', { username, password, session }),

  profileStatus: (username: string) =>
    get<ProfileStatus>(`/auth/profile/status?username=${encodeURIComponent(username)}`),
};
