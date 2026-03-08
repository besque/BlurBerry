/**
 * Backend API base URL for BlurBerry.
 * - When the app is served by the FastAPI server (e.g. python main.py), use '' (same origin).
 * - When using Vite dev server, relative URLs are proxied to the backend (see vite.config.ts).
 * - Set VITE_API_BASE in .env to point to another backend (e.g. http://127.0.0.1:8000).
 */
const API_BASE =
  typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE != null
    ? String(import.meta.env.VITE_API_BASE).replace(/\/$/, '')
    : '';

export function getApiBase(): string {
  return API_BASE;
}

/** Backend status response (GET /status) */
export interface StatusResponse {
  status: string;
  timestamp: number;
  pipeline_ready: boolean;
  audio_ready: boolean;
  server_stream_available?: boolean;
  config?: {
    blur_faces?: boolean;
    blur_plates?: boolean;
    blur_cards?: boolean;
    blur_nsfw?: boolean;
    blur_text_pii?: boolean;
    blur_strength?: number;
    detection_cadence?: number;
  };
}

export async function fetchStatus(): Promise<StatusResponse> {
  const r = await fetch(`${API_BASE}/status`);
  if (!r.ok) throw new Error(`Status ${r.status}`);
  return r.json();
}

export async function fetchConfigUpdate(body: Record<string, unknown>): Promise<void> {
  const r = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Config ${r.status}`);
}

export async function fetchProcessFrame(formData: FormData): Promise<Response> {
  return fetch(`${API_BASE}/process_frame`, { method: 'POST', body: formData });
}

export async function fetchEnroll(formData: FormData): Promise<Response> {
  return fetch(`${API_BASE}/enroll`, { method: 'POST', body: formData });
}

export async function fetchEnrollPreview(formData: FormData): Promise<Response> {
  return fetch(`${API_BASE}/enroll_preview`, { method: 'POST', body: formData });
}

export interface EnrollDetectResponse {
  faces: Array<{ bbox: number[] }>;
}

export async function fetchEnrollDetect(formData: FormData): Promise<EnrollDetectResponse> {
  const r = await fetch(`${API_BASE}/enroll/detect`, { method: 'POST', body: formData });
  if (!r.ok) throw new Error(`Enroll detect ${r.status}`);
  return r.json();
}

export async function fetchAudioEvents(): Promise<{ events: unknown[] }> {
  const r = await fetch(`${API_BASE}/audio/events`);
  if (!r.ok) return { events: [] };
  return r.json();
}

/** Server camera: latest processed frame (JPEG). 204 when no frame. */
export async function fetchStreamLatest(): Promise<Response> {
  return fetch(`${API_BASE}/stream/latest`);
}

/** Server camera: detection counts from last frame. */
export interface StreamStatsResponse {
  faces: number;
  plates: number;
  pii: number;
  nsfw: number;
}

export async function fetchStreamStats(): Promise<StreamStatsResponse> {
  const r = await fetch(`${API_BASE}/stream/stats`);
  if (!r.ok) return { faces: 0, plates: 0, pii: 0, nsfw: 0 };
  return r.json();
}
