"use client";

import { useEffect, useRef, useState } from "react";

import { SignalBars } from "./SignalBars";
import { TrackRow } from "./TrackRow";
import type { PlayContext } from "./PlayerProvider";
import { logEvent } from "@/lib/api";
import { getSessionId } from "@/lib/session";
import type { RecommendationResponse } from "@/lib/types";

/**
 * Renders a ranked list and logs one impression per track.
 *
 * The impression is what gives every arm a denominator: without it a variant
 * that returns tracks nobody plays looks identical to one that was never
 * served at all.
 */
export function ResultsList({ data }: { data: RecommendationResponse }) {
  const loggedRef = useRef<string>("");
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    // Guard against React strict-mode double-invocation and re-renders that
    // don't change the underlying result set.
    const key = `${data.variant}:${data.seed_track_id ?? data.query}:${data.results
      .map((r) => r.track.id)
      .join(",")}`;
    if (loggedRef.current === key || data.results.length === 0) return;
    loggedRef.current = key;

    data.results.forEach((rec, position) => {
      logEvent({
        session_id: getSessionId(),
        variant: data.variant,
        track_id: rec.track.id,
        seed_track_id: data.seed_track_id,
        action: "impression",
        position,
      });
    });
  }, [data]);

  const queue = data.results.map((rec, position) => ({
    track: rec.track,
    context: {
      variant: data.variant,
      seedTrackId: data.seed_track_id,
      position,
    } satisfies PlayContext,
  }));

  if (data.results.length === 0) {
    return (
      <p className="rounded-xl border border-white/10 px-4 py-10 text-center text-sm text-white/35">
        Nothing matched. Try a different description.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-0.5">
      {data.results.map((rec, i) => {
        const isOpen = expanded === rec.track.id;
        return (
          <div key={rec.track.id}>
            <div className="flex items-center">
              <div className="min-w-0 flex-1">
                <TrackRow
                  track={rec.track}
                  context={queue[i].context}
                  queue={queue}
                  signals={rec.signals}
                  index={i}
                />
              </div>
              <button
                onClick={() => setExpanded(isOpen ? null : rec.track.id)}
                aria-expanded={isOpen}
                aria-label={`${isOpen ? "Hide" : "Show"} why ${rec.track.name} was recommended`}
                className="shrink-0 rounded px-2 py-1 text-[10px] text-white/25 transition hover:bg-white/5 hover:text-white/70"
              >
                why {isOpen ? "▴" : "▾"}
              </button>
            </div>

            {isOpen && (
              <div className="mb-1 ml-9 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                <div className="mb-2 flex items-baseline justify-between">
                  <span className="text-[10px] uppercase tracking-wider text-white/35">
                    Why this track
                  </span>
                  <span className="font-mono text-[10px] text-white/25">
                    relevance {rec.relevance.toFixed(3)}
                  </span>
                </div>
                <SignalBars signals={rec.signals} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
