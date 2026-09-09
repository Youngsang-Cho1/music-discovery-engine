export interface Track {
  id: string;
  name: string;
  artist_name: string | null;
  album_name: string | null;
  duration_sec: number | null;
  audio_url: string | null;
  image_url: string | null;
  license_url: string | null;
  share_url: string | null;
  tags: string[];
  features: Record<string, number>;
}

/** Per-signal contributions behind one recommendation, used to explain it. */
export interface Signals {
  audio: number;
  descriptor: number;
  tag: number;
  novelty: number;
}

export interface Recommendation {
  track: Track;
  relevance: number;
  signals: Signals;
}

export interface RecommendationResponse {
  /** Bandit arm that produced this list. Echoed back on every event. */
  variant: string;
  session_id: string;
  seed_track_id: string | null;
  query: string | null;
  results: Recommendation[];
}

export type EventAction = "impression" | "play" | "skip" | "complete" | "like";

export interface EventPayload {
  session_id: string;
  variant: string;
  track_id: string;
  action: EventAction;
  seed_track_id?: string | null;
  position?: number | null;
  dwell_ms?: number | null;
}

export interface ArmStat {
  variant: string;
  impressions: number;
  rewards: number;
  posterior_mean: number;
  ci_low: number;
  ci_high: number;
  probability_best: number | null;
}

export interface ExperimentReport {
  arms: ArmStat[];
  total_impressions: number;
  total_rewards: number;
}

export interface TagCount {
  tag: string;
  count: number;
}

export interface CatalogStats {
  tracks: number;
  artists: number;
  total_duration_sec: number;
  top_tags: TagCount[];
}
