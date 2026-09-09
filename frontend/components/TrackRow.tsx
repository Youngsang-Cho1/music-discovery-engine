"use client";

import Link from "next/link";

import { Artwork } from "./Artwork";
import { usePlayer, type PlayContext } from "./PlayerProvider";
import { dominantSignal, formatDuration } from "@/lib/format";
import type { Signals, Track } from "@/lib/types";

interface Props {
  track: Track;
  context: PlayContext;
  queue?: { track: Track; context: PlayContext }[];
  signals?: Signals;
  index?: number;
}

/** Three bars that only animate while audio is actually playing, so the row
 *  distinguishes "current and playing" from "current but paused". */
function EqualizerBars() {
  return (
    <span aria-hidden className="flex h-3 items-end gap-[2px]">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-[2px] animate-[eq_0.9s_ease-in-out_infinite] rounded-sm bg-white/70"
          style={{ animationDelay: `${i * 0.15}s`, height: "100%" }}
        />
      ))}
    </span>
  );
}

export function TrackRow({ track, context, queue, signals, index }: Props) {
  const { current, isPlaying, play, toggle } = usePlayer();
  const isCurrent = current?.id === track.id;
  const why = dominantSignal(signals);

  return (
    <div
      className={`group relative flex items-center gap-3 rounded-lg px-2 py-2 transition-colors ${
        isCurrent ? "bg-white/[0.08]" : "hover:bg-white/[0.04]"
      }`}
    >
      {index !== undefined && (
        <span className="flex w-6 shrink-0 justify-end">
          {isCurrent && isPlaying ? (
            <EqualizerBars />
          ) : (
            <span
              className={`text-xs tabular-nums ${
                isCurrent ? "text-white/70" : "text-white/25"
              }`}
            >
              {index + 1}
            </span>
          )}
        </span>
      )}

      <div className="relative shrink-0">
        <Artwork src={track.image_url} alt={track.name} size={44} />
        <button
          onClick={() => (isCurrent ? toggle() : play(track, context, queue))}
          disabled={!track.audio_url}
          aria-label={isCurrent && isPlaying ? `Pause ${track.name}` : `Play ${track.name}`}
          className={`absolute inset-0 flex items-center justify-center rounded bg-black/55 text-sm text-white transition-opacity disabled:cursor-not-allowed ${
            isCurrent ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus:opacity-100"
          }`}
        >
          {isCurrent && isPlaying ? "❚❚" : "▶"}
        </button>
      </div>

      <div className="min-w-0 flex-1">
        <div className={`truncate text-sm ${isCurrent ? "text-white" : "text-white/90"}`}>
          {track.name}
        </div>
        <div className="flex items-center gap-1.5 truncate text-xs text-white/45">
          <span className="truncate">{track.artist_name ?? "Unknown artist"}</span>
          {why && (
            <>
              <span aria-hidden className="text-white/20">
                ·
              </span>
              <span className="shrink-0 text-white/35">{why}</span>
            </>
          )}
        </div>
      </div>

      <div className="hidden shrink-0 items-center gap-1 lg:flex">
        {track.tags.slice(0, 2).map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-white/[0.06] px-2 py-0.5 text-[10px] text-white/40"
          >
            {tag}
          </span>
        ))}
      </div>

      <span className="w-11 shrink-0 text-right text-xs text-white/35 tabular-nums">
        {formatDuration(track.duration_sec)}
      </span>

      <Link
        href={`/track/${track.id}`}
        title="Find tracks that sound like this"
        className="shrink-0 rounded px-2 py-1 text-xs text-white/35 opacity-0 transition hover:bg-white/10 hover:text-white focus:opacity-100 group-hover:opacity-100"
      >
        similar
      </Link>
    </div>
  );
}
