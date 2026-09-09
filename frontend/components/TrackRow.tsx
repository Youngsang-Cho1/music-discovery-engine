"use client";

import Link from "next/link";

import { usePlayer, type PlayContext } from "./PlayerProvider";
import type { Signals, Track } from "@/lib/types";

interface Props {
  track: Track;
  context: PlayContext;
  queue?: { track: Track; context: PlayContext }[];
  signals?: Signals;
  index?: number;
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return "--:--";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Which signal contributed most — the one-word "why this track" label. */
function dominantSignal(signals?: Signals): string | null {
  if (!signals) return null;
  const labels: Record<keyof Signals, string> = {
    audio: "similar sound",
    descriptor: "similar tempo & energy",
    tag: "shared genre",
    novelty: "hidden gem",
  };
  const entries = Object.entries(signals) as [keyof Signals, number][];
  const [key, value] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
  return value > 0 ? labels[key] : null;
}

export function TrackRow({ track, context, queue, signals, index }: Props) {
  const { current, isPlaying, play, toggle } = usePlayer();
  const isCurrent = current?.id === track.id;
  const why = dominantSignal(signals);

  return (
    <div
      className={`group flex items-center gap-3 rounded-lg px-3 py-2 transition-colors ${
        isCurrent ? "bg-white/10" : "hover:bg-white/5"
      }`}
    >
      <button
        onClick={() => (isCurrent ? toggle() : play(track, context, queue))}
        disabled={!track.audio_url}
        aria-label={isCurrent && isPlaying ? "Pause" : "Play"}
        className="flex size-10 shrink-0 items-center justify-center rounded-md bg-white/10 text-sm disabled:opacity-30"
      >
        {isCurrent && isPlaying ? "❚❚" : "▶"}
      </button>

      {index !== undefined && (
        <span className="w-5 shrink-0 text-right text-xs text-white/30 tabular-nums">
          {index + 1}
        </span>
      )}

      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{track.name}</div>
        <div className="truncate text-xs text-white/50">
          {track.artist_name ?? "Unknown artist"}
          {why && <span className="ml-2 text-white/30">· {why}</span>}
        </div>
      </div>

      <div className="hidden shrink-0 gap-1 sm:flex">
        {track.tags.slice(0, 2).map((tag) => (
          <span key={tag} className="rounded bg-white/5 px-2 py-0.5 text-[10px] text-white/40">
            {tag}
          </span>
        ))}
      </div>

      <span className="w-12 shrink-0 text-right text-xs text-white/40 tabular-nums">
        {formatDuration(track.duration_sec)}
      </span>

      <Link
        href={`/track/${track.id}`}
        className="shrink-0 rounded px-2 py-1 text-xs text-white/40 opacity-0 transition-opacity hover:text-white group-hover:opacity-100"
        title="Find similar tracks"
      >
        similar →
      </Link>
    </div>
  );
}
