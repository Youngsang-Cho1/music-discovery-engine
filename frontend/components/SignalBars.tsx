"use client";

import { SIGNAL_ORDER, signalLabel } from "@/lib/format";
import type { Signals } from "@/lib/types";

/**
 * Per-signal breakdown for one recommendation.
 *
 * The API returns all four contributions separately precisely so a ranking can
 * be explained rather than asserted. Collapsing them to a single "relevance"
 * number in the UI would throw that away — and it is the part that shows the
 * ranking is four weighted signals, not one opaque score.
 */
export function SignalBars({ signals }: { signals: Signals }) {
  return (
    <dl className="flex flex-col gap-1.5">
      {SIGNAL_ORDER.map((key) => {
        // Signals are already 0–1 (cosine, Jaccard, L1 similarity, damped
        // inverse popularity), so they share an axis without rescaling.
        const value = Math.max(0, Math.min(1, signals[key] ?? 0));
        return (
          <div key={key} className="flex items-center gap-2">
            <dt className="w-28 shrink-0 text-[10px] text-white/40">{signalLabel(key)}</dt>
            <dd className="flex flex-1 items-center gap-2">
              <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/[0.07]">
                <div
                  className="h-full rounded-full bg-white/45"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
              <span className="w-8 shrink-0 text-right text-[10px] text-white/30 tabular-nums">
                {value.toFixed(2)}
              </span>
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
