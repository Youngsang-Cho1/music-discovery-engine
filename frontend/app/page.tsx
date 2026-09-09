"use client";

import { useCallback, useEffect, useState } from "react";

import { ResultsList } from "@/components/ResultsList";
import { ResultsSkeleton } from "@/components/Skeleton";
import { TrackRow } from "@/components/TrackRow";
import { ApiError, randomTracks, recommendByText } from "@/lib/api";
import type { RecommendationResponse, Track } from "@/lib/types";

// These work only because audio and text share one embedding space — none is a
// genre tag, and none would match anything by title.
const EXAMPLES = [
  "rainy day lo-fi with soft piano",
  "driving synth at night",
  "warm acoustic fingerpicking",
  "tense cinematic strings",
  "distorted guitar, fast and angry",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RecommendationResponse | null>(null);
  const [seeds, setSeeds] = useState<Track[]>([]);
  const [loading, setLoading] = useState(false);
  const [seedsLoading, setSeedsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Text search being off on a deployment is not an error — the rest of the
  // app works — so it gets a notice rather than a red failure box.
  const [unavailable, setUnavailable] = useState<string | null>(null);

  useEffect(() => {
    randomTracks(10)
      .then(setSeeds)
      .catch(() => setSeeds([]))
      .finally(() => setSeedsLoading(false));
  }, []);

  const search = useCallback(async (q: string) => {
    if (q.trim().length < 2) return;
    setLoading(true);
    setError(null);
    setUnavailable(null);
    try {
      setResults(await recommendByText(q, { topK: 20 }));
    } catch (err) {
      setResults(null);
      if (err instanceof ApiError && err.isUnavailable) setUnavailable(err.message);
      else setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="flex flex-col gap-12">
      <section className="flex flex-col gap-5">
        <div className="max-w-2xl">
          <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
            Describe a sound.
            <br />
            <span className="text-white/50">Get music that sounds like it.</span>
          </h1>
          <p className="mt-3 text-sm leading-relaxed text-white/50">
            Every track is Creative-Commons licensed and matched on its audio, not its play
            count — so a track with three plays can outrank one with a million.
          </p>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            void search(query);
          }}
          className="flex flex-col gap-2 sm:flex-row"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="rainy day lo-fi with soft piano"
            aria-label="Describe the sound you want"
            className="flex-1 rounded-xl border border-white/15 bg-white/[0.04] px-4 py-3.5 text-sm outline-none transition placeholder:text-white/25 focus:border-white/40 focus:bg-white/[0.06]"
          />
          <button
            type="submit"
            disabled={loading || query.trim().length < 2}
            className="rounded-xl bg-white px-6 py-3.5 text-sm font-medium text-black transition hover:bg-white/90 disabled:opacity-30"
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
              className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-white/55 transition hover:border-white/30 hover:text-white"
            >
              {example}
            </button>
          ))}
        </div>

        {error && (
          <p className="rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </p>
        )}

        {unavailable && (
          <p className="rounded-xl border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
            {unavailable} Pick any track below and choose{" "}
            <span className="font-medium">similar</span> instead.
          </p>
        )}
      </section>

      {loading && (
        <section className="flex flex-col gap-3">
          <h2 className="text-xs uppercase tracking-wider text-white/35">Searching</h2>
          <ResultsSkeleton />
        </section>
      )}

      {!loading && results && (
        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm text-white/70">
              Sounds like{" "}
              <span className="text-white">&ldquo;{results.query}&rdquo;</span>
            </h2>
            <span
              title="Ranking variant chosen by the Thompson-sampling bandit"
              className="font-mono text-[10px] text-white/25"
            >
              arm: {results.variant}
            </span>
          </div>
          <ResultsList data={results} />
        </section>
      )}

      {/* The seed list stays visible when text search is off — it is the
          fallback the notice above points at. */}
      {(!results || unavailable) && !loading && (
        <section className="flex flex-col gap-3">
          <h2 className="text-xs uppercase tracking-wider text-white/35">
            Or start from a track
          </h2>
          {seedsLoading ? (
            <ResultsSkeleton rows={5} />
          ) : seeds.length > 0 ? (
            <div className="flex flex-col gap-0.5">
              {seeds.map((track, i) => (
                <TrackRow
                  key={track.id}
                  track={track}
                  index={i}
                  context={{ variant: "browse" }}
                  queue={seeds.map((t) => ({ track: t, context: { variant: "browse" } }))}
                />
              ))}
            </div>
          ) : (
            <p className="rounded-xl border border-white/10 px-4 py-8 text-center text-sm text-white/35">
              Catalog is empty. Run the ingestion pipeline to populate it.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
