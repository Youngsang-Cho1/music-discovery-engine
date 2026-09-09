"use client";

import { useCallback, useEffect, useState } from "react";

import { ResultsList } from "@/components/ResultsList";
import { TrackRow } from "@/components/TrackRow";
import { randomTracks, recommendByText } from "@/lib/api";
import type { RecommendationResponse, Track } from "@/lib/types";

// Prompts that only work because audio and text share one embedding space —
// none of these are genre tags, and none would match on track titles.
const EXAMPLES = [
  "rainy day lo-fi with soft piano",
  "driving synth at night",
  "warm acoustic fingerpicking",
  "tense cinematic strings",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RecommendationResponse | null>(null);
  const [seeds, setSeeds] = useState<Track[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    randomTracks(10)
      .then(setSeeds)
      .catch(() => setSeeds([]));
  }, []);

  const search = useCallback(async (q: string) => {
    if (q.trim().length < 2) return;
    setLoading(true);
    setError(null);
    try {
      setResults(await recommendByText(q, { topK: 20 }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="flex flex-col gap-10">
      <section className="flex flex-col gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Describe a sound. Get music that sounds like it.
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-white/50">
            Every track here is Creative-Commons licensed and matched on its audio, not its
            play count — so a track with three plays can outrank one with a million.
          </p>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            void search(query);
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. rainy day lo-fi with soft piano"
            aria-label="Describe the sound you want"
            className="flex-1 rounded-lg border border-white/15 bg-white/5 px-4 py-3 text-sm outline-none placeholder:text-white/30 focus:border-white/40"
          />
          <button
            type="submit"
            disabled={loading || query.trim().length < 2}
            className="rounded-lg bg-white px-5 py-3 text-sm font-medium text-black disabled:opacity-40"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </form>

        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => {
                setQuery(example);
                void search(example);
              }}
              className="rounded-full border border-white/15 px-3 py-1 text-xs text-white/60 hover:border-white/40 hover:text-white"
            >
              {example}
            </button>
          ))}
        </div>

        {error && (
          <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </p>
        )}
      </section>

      {results && (
        <section className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium text-white/70">
              Results for &ldquo;{results.query}&rdquo;
            </h2>
            <span className="font-mono text-[10px] text-white/25">arm: {results.variant}</span>
          </div>
          <ResultsList data={results} />
        </section>
      )}

      {!results && seeds.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-medium text-white/70">Or start from a track</h2>
          <div className="flex flex-col gap-1">
            {seeds.map((track) => (
              <TrackRow
                key={track.id}
                track={track}
                context={{ variant: "browse" }}
                queue={seeds.map((t) => ({ track: t, context: { variant: "browse" } }))}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
