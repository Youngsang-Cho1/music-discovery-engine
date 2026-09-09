"use client";

import { useEffect, useRef } from "react";

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
    return <p className="py-8 text-center text-sm text-white/40">No matches yet.</p>;
  }

  return (
    <div className="flex flex-col gap-1">
      {data.results.map((rec, i) => (
        <TrackRow
          key={rec.track.id}
          track={rec.track}
          context={queue[i].context}
          queue={queue}
          signals={rec.signals}
          index={i}
        />
      ))}
    </div>
  );
}
