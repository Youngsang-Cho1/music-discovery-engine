"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";

import { Artwork } from "@/components/Artwork";
import { FeatureChips } from "@/components/FeatureChips";
import { ResultsList } from "@/components/ResultsList";
import { ResultsSkeleton, Skeleton } from "@/components/Skeleton";
import { usePlayer } from "@/components/PlayerProvider";
import { getTrack, recommendByTrack } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import type { RecommendationResponse, Track } from "@/lib/types";

export default function TrackPage({ params }: PageProps<"/track/[id]">) {
  // Route params are a promise in this version of Next; `use` unwraps it.
  const { id } = use(params);
  const { play, current, isPlaying, toggle } = usePlayer();

  // One state object tagged with the id it belongs to. Resetting three
  // separate pieces of state inside the effect would cascade renders, and
  // stale data would flash between navigations.
  const [loaded, setLoaded] = useState<{
    id: string;
    seed: Track | null;
    results: RecommendationResponse | null;
    error: string | null;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;

    Promise.all([getTrack(id), recommendByTrack(id, { topK: 20 })])
      .then(([track, recs]) => {
        if (!cancelled) setLoaded({ id, seed: track, results: recs, error: null });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setLoaded({
            id,
            seed: null,
            results: null,
            error: err instanceof Error ? err.message : "Failed to load",
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id]);

  // Anything tagged with a previous id is stale while the new one loads.
  const fresh = loaded?.id === id ? loaded : null;
  const seed = fresh?.seed ?? null;
  const results = fresh?.results ?? null;
  const error = fresh?.error ?? null;

  if (error) {
    return (
      <div className="flex flex-col gap-4">
        <p className="rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </p>
        <Link href="/" className="text-sm text-white/50 hover:text-white">
          ← Back to search
        </Link>
      </div>
    );
  }

  const isCurrent = seed && current?.id === seed.id;

  return (
    <div className="flex flex-col gap-10">
      <section className="flex flex-col gap-6">
        <Link href="/" className="w-fit text-xs text-white/40 transition hover:text-white">
          ← Back to search
        </Link>

        {!seed ? (
          <div className="flex gap-5">
            <Skeleton className="size-32 shrink-0" />
            <div className="flex flex-1 flex-col gap-3 pt-2">
              <Skeleton className="h-6 w-1/2" />
              <Skeleton className="h-4 w-1/3" />
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-5 sm:flex-row sm:gap-6">
            <Artwork src={seed.image_url} alt={seed.name} size={128} className="rounded-lg" />

            <div className="flex min-w-0 flex-1 flex-col gap-3">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-white/35">
                  Seed track
                </div>
                <h1 className="mt-1 text-balance text-2xl font-semibold tracking-tight">
                  {seed.name}
                </h1>
                <p className="text-sm text-white/50">
                  {seed.artist_name ?? "Unknown artist"}
                  {seed.album_name && ` · ${seed.album_name}`}
                  {seed.duration_sec ? ` · ${formatDuration(seed.duration_sec)}` : ""}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() =>
                    isCurrent ? toggle() : play(seed, { variant: "browse" })
                  }
                  disabled={!seed.audio_url}
                  className="rounded-full bg-white px-5 py-2 text-sm font-medium text-black transition hover:bg-white/90 disabled:opacity-30"
                >
                  {isCurrent && isPlaying ? "❚❚ Pause" : "▶ Play"}
                </button>
                {seed.tags.slice(0, 4).map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full bg-white/[0.06] px-2.5 py-1 text-[11px] text-white/45"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {seed && Object.keys(seed.features).length > 0 && (
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
            <h2 className="mb-3 text-[10px] uppercase tracking-wider text-white/35">
              Audio profile
            </h2>
            {/* Same numbers the `descriptor` ranking signal uses, so the
                "similar tempo & energy" explanation below is checkable. */}
            <FeatureChips features={seed.features} />
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm text-white/70">Sounds like this</h2>
          {results && (
            <span
              title="Ranking variant chosen by the Thompson-sampling bandit"
              className="font-mono text-[10px] text-white/25"
            >
              arm: {results.variant}
            </span>
          )}
        </div>
        {results ? <ResultsList data={results} /> : <ResultsSkeleton />}
      </section>
    </div>
  );
}
