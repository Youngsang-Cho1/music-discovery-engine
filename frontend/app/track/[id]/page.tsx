"use client";

import { use, useEffect, useState } from "react";

import { ResultsList } from "@/components/ResultsList";
import { TrackRow } from "@/components/TrackRow";
import { getTrack, recommendByTrack } from "@/lib/api";
import type { RecommendationResponse, Track } from "@/lib/types";

export default function TrackPage({ params }: PageProps<"/track/[id]">) {
  // Route params are a promise in this version of Next; `use` unwraps it.
  const { id } = use(params);

  const [seed, setSeed] = useState<Track | null>(null);
  const [results, setResults] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([getTrack(id), recommendByTrack(id, { topK: 20 })])
      .then(([track, recs]) => {
        if (cancelled) return;
        setSeed(track);
        setResults(recs);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load");
      });

    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    return (
      <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
        {error}
      </p>
    );
  }

  if (!seed || !results) {
    return <p className="text-sm text-white/40">Loading…</p>;
  }

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-3">
        <span className="text-xs uppercase tracking-wider text-white/40">Seed track</span>
        <TrackRow track={seed} context={{ variant: "browse" }} />
        {Object.keys(seed.features).length > 0 && (
          <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-white/40">
            {Object.entries(seed.features).map(([key, value]) => (
              <div key={key} className="flex gap-1">
                <dt>{key.replace(/_/g, " ")}</dt>
                <dd className="text-white/60 tabular-nums">{value.toFixed(1)}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-medium text-white/70">Sounds like this</h2>
          <span className="font-mono text-[10px] text-white/25">arm: {results.variant}</span>
        </div>
        <ResultsList data={results} />
      </section>
    </div>
  );
}
