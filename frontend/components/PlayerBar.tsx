"use client";

import Link from "next/link";

import { Artwork } from "./Artwork";
import { usePlayer } from "./PlayerProvider";
import { formatDuration } from "@/lib/format";

export function PlayerBar() {
  const { current, isPlaying, progress, duration, toggle, next, seek, like, liked } =
    usePlayer();

  if (!current) return null;

  const pct = duration > 0 ? (progress / duration) * 100 : 0;

  return (
    <div className="fixed inset-x-0 bottom-0 z-50 border-t border-white/10 bg-neutral-950/85 backdrop-blur-xl">
      {/* Progress as a hairline across the whole bar, so position is readable
          at a glance without hunting for a slider. */}
      <div className="h-0.5 w-full bg-white/10">
        <div className="h-full bg-white/70 transition-[width]" style={{ width: `${pct}%` }} />
      </div>

      <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
        <button
          onClick={toggle}
          aria-label={isPlaying ? "Pause" : "Play"}
          className="flex size-10 shrink-0 items-center justify-center rounded-full bg-white text-sm text-black transition hover:scale-105"
        >
          {isPlaying ? "❚❚" : "▶"}
        </button>
        <button
          onClick={next}
          aria-label="Next track"
          className="shrink-0 px-1 text-white/45 transition hover:text-white"
        >
          ⏭
        </button>

        <Artwork src={current.image_url} alt={current.name} size={40} />

        <div className="min-w-0 flex-1">
          <div className="truncate text-sm">{current.name}</div>
          <div className="truncate text-xs text-white/45">{current.artist_name}</div>
        </div>

        <span className="hidden shrink-0 text-xs text-white/40 tabular-nums sm:block">
          {formatDuration(progress)} / {formatDuration(duration)}
        </span>

        <input
          type="range"
          min={0}
          max={duration || 0}
          value={progress}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Seek"
          className="hidden h-1 w-32 shrink-0 accent-white md:block"
        />

        <button
          onClick={like}
          aria-pressed={liked}
          title="Counts as a positive outcome for the ranking arm that served this track"
          className={`shrink-0 rounded-full border px-3 py-1 text-xs transition ${
            liked
              ? "border-rose-400/40 bg-rose-400/15 text-rose-200"
              : "border-white/15 text-white/70 hover:bg-white/10"
          }`}
        >
          {liked ? "♥" : "♡"} Like
        </button>

        <Link
          href={`/track/${current.id}`}
          className="hidden shrink-0 text-xs text-white/35 transition hover:text-white sm:block"
        >
          similar
        </Link>

        {/* Shortcuts are worthless if nobody knows they exist. */}
        <span
          title="Space play/pause · ← → seek 10s · N next · L like"
          className="hidden shrink-0 cursor-help font-mono text-[10px] text-white/20 xl:block"
        >
          ␣ ← → N L
        </span>

        {/* Attribution is a licence condition for the CC tracks we stream. */}
        {current.license_url && (
          <a
            href={current.share_url ?? current.license_url}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden shrink-0 text-[10px] text-white/25 transition hover:text-white/60 lg:block"
          >
            CC · Jamendo
          </a>
        )}
      </div>
    </div>
  );
}
