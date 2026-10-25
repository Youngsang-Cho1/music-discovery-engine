import type { Signals } from "./types";

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || !Number.isFinite(seconds)) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Which signal contributed most — the one-line "why this track" label. */
const SIGNAL_LABELS: Record<keyof Signals, string> = {
  audio: "similar sound",
  descriptor: "similar tempo & energy",
  tag: "shared genre",
  novelty: "hidden gem",
};

export function dominantSignal(signals?: Signals): string | null {
  if (!signals) return null;
  const entries = Object.entries(signals) as [keyof Signals, number][];
  const [key, value] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
  return value > 0 ? SIGNAL_LABELS[key] : null;
}

export const SIGNAL_ORDER: (keyof Signals)[] = ["audio", "descriptor", "tag", "novelty"];

export function signalLabel(key: keyof Signals): string {
  return SIGNAL_LABELS[key];
}

/**
 * Turn a raw descriptor into something a listener can read.
 *
 * The API returns physical units — 3233 Hz of spectral centroid means nothing
 * to anyone. Each descriptor is mapped onto its conventional range and given a
 * word, so the UI can explain a match instead of just asserting it.
 */
export interface ReadableFeature {
  key: string;
  label: string;
  value: string;
  /** 0–1, for bar widths. */
  fraction: number;
}

const RANGES: Record<string, [number, number]> = {
  tempo_bpm: [40, 220],
  loudness_db: [-60, 0],
  brightness: [0, 8000],
  percussiveness: [0, 1],
  dynamic_range: [0, 0.3],
  onset_density: [0, 8],
};

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function bucket(fraction: number, words: [string, string, string]): string {
  if (fraction < 0.34) return words[0];
  if (fraction < 0.67) return words[1];
  return words[2];
}

export function readableFeatures(features: Record<string, number>): ReadableFeature[] {
  const out: ReadableFeature[] = [];

  for (const [key, raw] of Object.entries(features)) {
    const [low, high] = RANGES[key] ?? [0, 1];
    const fraction = clamp01((raw - low) / (high - low));

    switch (key) {
      case "tempo_bpm":
        out.push({ key, label: "Tempo", value: `${Math.round(raw)} BPM`, fraction });
        break;
      case "loudness_db":
        out.push({
          key,
          label: "Loudness",
          value: bucket(fraction, ["quiet", "moderate", "loud"]),
          fraction,
        });
        break;
      case "brightness":
        out.push({
          key,
          label: "Brightness",
          value: bucket(fraction, ["dark", "warm", "bright"]),
          fraction,
        });
        break;
      case "percussiveness":
        out.push({
          key,
          label: "Texture",
          value: bucket(fraction, ["smooth", "mixed", "percussive"]),
          fraction,
        });
        break;
      case "dynamic_range":
        out.push({
          key,
          label: "Dynamics",
          value: bucket(fraction, ["steady", "varied", "dramatic"]),
          fraction,
        });
        break;
      case "onset_density":
        out.push({
          key,
          label: "Density",
          value: bucket(fraction, ["sparse", "moderate", "busy"]),
          fraction,
        });
        break;
    }
  }

  // Stable order regardless of JSON key order.
  const order = Object.keys(RANGES);
  return out.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
}

/**
 * Compact play count: 1_672_904 -> "1.7M".
 *
 * Exact counts are noise at this scale — nobody reads the difference between
 * 1,672,904 and 1,681,220 — and the column has to stay narrow enough to sit
 * next to the duration without pushing the title out.
 */
export function formatPlays(listens: number | null | undefined): string {
  const n = listens ?? 0;
  if (!Number.isFinite(n) || n < 0) return "0";
  if (n < 1_000) return String(Math.floor(n));
  if (n < 1_000_000) {
    const k = n / 1_000;
    return `${k < 10 ? k.toFixed(1) : Math.round(k)}K`;
  }
  const m = n / 1_000_000;
  return `${m < 10 ? m.toFixed(1) : Math.round(m)}M`;
}

/**
 * How far a track has travelled, as three buckets the UI styles differently.
 *
 * `unheard` is the load-bearing one. The `novelty` signal is defined against
 * play count, and the `discovery` arm exists to surface tracks the popularity
 * ranking never would. Without a visible marker, that arm's results look
 * identical to any other list and the whole claim is invisible to the person
 * using it — which is how the signal stayed broken for as long as it did.
 */
export type ReachTier = "unheard" | "deep-cut" | "popular";

export function reachTier(listens: number | null | undefined): ReachTier {
  const n = listens ?? 0;
  if (n <= 0) return "unheard";
  if (n < 1_000) return "deep-cut";
  return "popular";
}
