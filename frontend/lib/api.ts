import { getSessionId } from "./session";
import type {
  EventPayload,
  ExperimentReport,
  RecommendationResponse,
  Track,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** An error carrying the HTTP status, so callers can distinguish a feature
 *  being switched off (503) from a genuine failure. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** Text search is unavailable on this deployment — not a bug. */
  get isUnavailable(): boolean {
    return this.status === 503;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, "Can't reach the API. Is the backend running?");
  }

  if (!res.ok) {
    // FastAPI puts the human-readable reason in `detail`.
    const detail = await res
      .json()
      .then((body) => body?.detail)
      .catch(() => null);
    throw new ApiError(res.status, detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export interface RecommendFilters {
  tags?: string[];
  tempoMin?: number;
  tempoMax?: number;
  topK?: number;
  /** Pin a specific arm. Used by the compare view, not by normal traffic. */
  forceVariant?: string;
}

function filterParams(filters: RecommendFilters = {}): URLSearchParams {
  const params = new URLSearchParams();
  params.set("session_id", getSessionId());
  if (filters.topK) params.set("top_k", String(filters.topK));
  if (filters.tempoMin !== undefined) params.set("tempo_min", String(filters.tempoMin));
  if (filters.tempoMax !== undefined) params.set("tempo_max", String(filters.tempoMax));
  if (filters.forceVariant) params.set("force_variant", filters.forceVariant);
  // FastAPI reads repeated keys as a list.
  filters.tags?.forEach((tag) => params.append("tags", tag));
  return params;
}

export function listTracks(q?: string, limit = 50): Promise<Track[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (q) params.set("q", q);
  return request<Track[]>(`/tracks?${params}`);
}

export function randomTracks(limit = 12): Promise<Track[]> {
  return request<Track[]>(`/tracks/random?limit=${limit}`);
}

export function getTrack(id: string): Promise<Track> {
  return request<Track>(`/tracks/${id}`);
}

export function recommendByTrack(
  trackId: string,
  filters?: RecommendFilters,
): Promise<RecommendationResponse> {
  return request<RecommendationResponse>(
    `/recommend/track/${trackId}?${filterParams(filters)}`,
  );
}

export function recommendByText(
  query: string,
  filters?: RecommendFilters,
): Promise<RecommendationResponse> {
  const params = filterParams(filters);
  params.set("q", query);
  return request<RecommendationResponse>(`/recommend/text?${params}`);
}

export function getExperimentReport(): Promise<ExperimentReport> {
  return request<ExperimentReport>("/experiments/report");
}

/**
 * Fire-and-forget event logging.
 *
 * Uses sendBeacon when the page is unloading — a normal fetch is cancelled
 * mid-flight on navigation, which would silently drop exactly the skip events
 * the bandit most needs.
 */
export function logEvent(event: EventPayload, opts: { beacon?: boolean } = {}): void {
  const url = `${BASE_URL}/events`;
  const body = JSON.stringify(event);

  if (opts.beacon && typeof navigator !== "undefined" && navigator.sendBeacon) {
    navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
    return;
  }

  void fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {
    // Telemetry must never surface as a user-visible error.
  });
}
