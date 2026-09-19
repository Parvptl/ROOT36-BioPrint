/** Response shapes from the BioPrint API. Mirrors backend/app/models/schemas.py. */

export interface Challenge {
  nonce: string;
  phrase: string;
  expires_in: number;
  round_index: number | null;
  rounds_total: number | null;
}

export interface EnrollmentProgress {
  accepted: boolean;
  sessions_captured: number;
  sessions_required: number;
  profile_built: boolean;
  message: string;
  quality: Record<string, unknown> | null;
}

export interface SignalDetail {
  code: string;
  label: string;
  band: 'LOW' | 'MEDIUM' | 'HIGH';
  detail: string;
}

export interface LatencyBreakdown {
  total_ms: number;
  validation_ms: number;
  extraction_ms: number;
  identity_ms: number;
  automation_ms: number;
}

export interface Decision {
  decision: 'ALLOW' | 'BLOCK';
  reason: string;
  message: string;
  identity_score: number | null;
  automation_score: number | null;
  integrity_score: number;
  coverage: number | null;
  threshold: number | null;
  modality_scores: Record<string, number> | null;
  signals: SignalDetail[];
  latency: LatencyBreakdown;
  session_token: string | null;
  attempt_id: number | null;
}

export interface ProfileStatus {
  username: string;
  enrolled: boolean;
  sessions_captured: number;
  sessions_required: number;
  feature_count: number | null;
  population_size: number | null;
  threshold: number | null;
  threshold_source: string | null;
  calibration_note: string | null;
}

export interface RegisterResult {
  user_id: number;
  username: string;
  enrolled: boolean;
  sessions_required: number;
}
