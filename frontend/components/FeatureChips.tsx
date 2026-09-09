"use client";

import { readableFeatures } from "@/lib/format";

/**
 * Audio descriptors, translated.
 *
 * The API returns physical units — 3233 Hz of spectral centroid is meaningless
 * to a listener. These are the same numbers the `descriptor` ranking signal
 * uses, rendered as words so the "similar tempo & energy" explanation on a
 * result is checkable rather than something the UI just claims.
 */
export function FeatureChips({ features }: { features: Record<string, number> }) {
  const readable = readableFeatures(features);
  if (readable.length === 0) return null;

  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
      {readable.map((f) => (
        <div key={f.key} className="flex flex-col gap-1">
          <div className="flex items-baseline justify-between gap-2">
            <dt className="text-[10px] uppercase tracking-wide text-white/35">{f.label}</dt>
            <dd className="text-xs text-white/80">{f.value}</dd>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-white/[0.07]">
            <div
              className="h-full rounded-full bg-white/30"
              style={{ width: `${f.fraction * 100}%` }}
            />
          </div>
        </div>
      ))}
    </dl>
  );
}
