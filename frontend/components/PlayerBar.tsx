"use client";

import { usePlayer } from "./PlayerProvider";

function fmt(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function PlayerBar() {
  const { current, isPlaying, progress, duration, toggle, next, seek, like } = usePlayer();

  if (!current) return null;

  return (
    <div className="fixed inset-x-0 bottom-0 z-50 border-t border-white/10 bg-black/80 backdrop-blur-lg">
      <div className="mx-auto flex max-w-5xl items-center gap-4 px-4 py-3">
        <button
          onClick={toggle}
          aria-label={isPlaying ? "Pause" : "Play"}
          className="flex size-10 shrink-0 items-center justify-center rounded-full bg-white text-black"
        >
          {isPlaying ? "❚❚" : "▶"}
        </button>
        <button
          onClick={next}
          aria-label="Next track"
          className="shrink-0 text-white/50 hover:text-white"
        >
          ⏭
        </button>

        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{current.name}</div>
          <div className="truncate text-xs text-white/50">{current.artist_name}</div>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-[10px] text-white/40 tabular-nums">{fmt(progress)}</span>
            <input
              type="range"
              min={0}
              max={duration || 0}
              value={progress}
              onChange={(e) => seek(Number(e.target.value))}
              aria-label="Seek"
              className="h-1 flex-1 accent-white"
            />
            <span className="text-[10px] text-white/40 tabular-nums">{fmt(duration)}</span>
          </div>
        </div>

        <button
          onClick={like}
          className="shrink-0 rounded-full border border-white/20 px-3 py-1 text-xs hover:bg-white/10"
        >
          ♥ Like
        </button>

        {/* Attribution is a licence condition for the CC tracks we stream. */}
        {current.license_url && (
          <a
            href={current.share_url ?? current.license_url}
            target="_blank"
            rel="noopener noreferrer"
            className="hidden shrink-0 text-[10px] text-white/30 hover:text-white/60 sm:block"
          >
            CC · Jamendo
          </a>
        )}
      </div>
    </div>
  );
}
