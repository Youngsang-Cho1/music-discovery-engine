"use client";

import { useEffect, useState } from "react";

import { ResultsSkeleton } from "@/components/Skeleton";
import { TrackRow } from "@/components/TrackRow";
import { getCatalogStats, listTracks } from "@/lib/api";
import type { CatalogStats, Track } from "@/lib/types";

/** Descriptive tags Jamendo attaches that aren't useful as browse filters. */
const NOT_A_GENRE = new Set(["voice", "vocal", "neutral", "acoustic", "instrument"]);

export default function BrowsePage() {
  const [stats, setStats] = useState<CatalogStats | null>(null);
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [activeTag, setActiveTag] = useState<string | null>(null);

  // Results tagged with the request they belong to, so a stale response from a
  // slower earlier request can't overwrite a newer one, and nothing has to be
  // reset synchronously inside the effect.
  const requestKey = `${submitted}::${activeTag ?? ""}`;
  const [loaded, setLoaded] = useState<{ key: string; tracks: Track[] } | null>(null);

  useEffect(() => {
    getCatalogStats()
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    // Filtering server-side, not over the loaded page: the client only holds
    // 100 rows, so filtering here would report "none" for tags that have
    // hundreds of tracks further down the catalog.
    listTracks({
      q: submitted || undefined,
      tags: activeTag ? [activeTag] : undefined,
      limit: 100,
    })
      .then((rows) => {
        if (!cancelled) setLoaded({ key: requestKey, tracks: rows });
      })
      .catch(() => {
        if (!cancelled) setLoaded({ key: requestKey, tracks: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [requestKey, submitted, activeTag]);

  const visible = loaded?.key === requestKey ? loaded.tracks : null;

  const genreTags = (stats?.top_tags ?? []).filter((t) => !NOT_A_GENRE.has(t.tag)).slice(0, 14);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Browse the catalog</h1>
        {stats ? (
          <p className="mt-2 text-sm text-white/50">
            <span className="text-white">{stats.tracks.toLocaleString()}</span> tracks from{" "}
            <span className="text-white">{stats.artists.toLocaleString()}</span> artists ·{" "}
            {Math.round(stats.total_duration_sec / 3600).toLocaleString()} hours of
            Creative-Commons audio
          </p>
        ) : (
          <p className="mt-2 text-sm text-white/30">Loading catalog…</p>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          setSubmitted(query);
        }}
        className="flex gap-2"
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by track or artist name"
          aria-label="Search by track or artist name"
          className="flex-1 rounded-xl border border-white/15 bg-white/[0.04] px-4 py-3 text-sm outline-none transition placeholder:text-white/25 focus:border-white/40"
        />
        <button
          type="submit"
          className="rounded-xl border border-white/15 px-5 py-3 text-sm transition hover:bg-white/10"
        >
          Search
        </button>
      </form>

      {/* Name search only — semantic search lives on the home page and goes
          through CLAP rather than SQL LIKE. */}
      <p className="-mt-6 text-xs text-white/30">
        Exact name match. For &ldquo;music that sounds like X&rdquo;, use the search on the
        home page.
      </p>

      {genreTags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveTag(null)}
            className={`rounded-full px-3 py-1.5 text-xs transition ${
              activeTag === null
                ? "bg-white text-black"
                : "border border-white/10 text-white/50 hover:border-white/30 hover:text-white"
            }`}
          >
            All
          </button>
          {genreTags.map((t) => (
            <button
              key={t.tag}
              onClick={() => setActiveTag(activeTag === t.tag ? null : t.tag)}
              className={`rounded-full px-3 py-1.5 text-xs transition ${
                activeTag === t.tag
                  ? "bg-white text-black"
                  : "border border-white/10 text-white/50 hover:border-white/30 hover:text-white"
              }`}
            >
              {t.tag}
              <span className="ml-1.5 opacity-50">{t.count}</span>
            </button>
          ))}
        </div>
      )}

      {visible === null ? (
        <ResultsSkeleton rows={8} />
      ) : visible.length === 0 ? (
        <p className="rounded-xl border border-white/10 px-4 py-10 text-center text-sm text-white/35">
          {activeTag ? `No tracks tagged “${activeTag}”.` : "No tracks matched."}
        </p>
      ) : (
        <div className="flex flex-col gap-0.5">
          {visible.map((track, i) => (
            <TrackRow
              key={track.id}
              track={track}
              index={i}
              context={{ variant: "browse" }}
              queue={visible.map((t) => ({ track: t, context: { variant: "browse" } }))}
            />
          ))}
        </div>
      )}
    </div>
  );
}
