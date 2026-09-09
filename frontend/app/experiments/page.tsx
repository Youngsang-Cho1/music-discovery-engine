"use client";

import { useEffect, useState } from "react";

import { Skeleton } from "@/components/Skeleton";
import { getExperimentReport } from "@/lib/api";
import type { ArmStat, ExperimentReport } from "@/lib/types";

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

/** What each arm is actually testing, so the table isn't four opaque names. */
const ARM_DESCRIPTIONS: Record<string, string> = {
  control: "v1 parity — audio similarity only, no diversification",
  descriptor_heavy: "leans on tempo and energy continuity",
  diverse: "aggressive MMR, prioritises variety over closeness",
  discovery: "rewards low-play-count tracks — the long-tail bet",
};

/**
 * Credible interval drawn to a shared scale.
 *
 * A wide bar means the arm has not earned a verdict yet. Showing only the
 * posterior mean would hide exactly that, which is the failure mode this
 * dashboard exists to prevent.
 */
function IntervalBar({ arm, scale, leading }: { arm: ArmStat; scale: number; leading: boolean }) {
  const left = (arm.ci_low / scale) * 100;
  const width = ((arm.ci_high - arm.ci_low) / scale) * 100;
  const mean = (arm.posterior_mean / scale) * 100;

  return (
    <div className="relative h-2.5 rounded-full bg-white/[0.05]">
      <div
        className={`absolute h-2.5 rounded-full ${leading ? "bg-emerald-400/30" : "bg-white/20"}`}
        style={{ left: `${left}%`, width: `${Math.max(width, 0.5)}%` }}
      />
      <div
        className={`absolute h-2.5 w-0.5 ${leading ? "bg-emerald-300" : "bg-white/70"}`}
        style={{ left: `${mean}%` }}
      />
    </div>
  );
}

export default function ExperimentsPage() {
  const [report, setReport] = useState<ExperimentReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      getExperimentReport()
        .then(setReport)
        .catch((err: unknown) =>
          setError(err instanceof Error ? err.message : "Failed to load"),
        );

    void load();
    const timer = setInterval(load, 10_000);
    return () => clearInterval(timer);
  }, []);

  if (error) {
    return (
      <p className="rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-300">
        {error}
      </p>
    );
  }

  if (!report) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-7 w-64" />
        <Skeleton className="h-16 w-full" />
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
    );
  }

  const scale = Math.max(...report.arms.map((a) => a.ci_high), 0.01);
  const leader = report.arms.reduce((a, b) =>
    (b.probability_best ?? 0) > (a.probability_best ?? 0) ? b : a,
  );
  const noData = report.total_impressions === 0;
  const rate = report.total_impressions
    ? report.total_rewards / report.total_impressions
    : 0;

  return (
    <div className="flex flex-col gap-8">
      <div className="max-w-2xl">
        <h1 className="text-2xl font-semibold tracking-tight">Live ranking experiment</h1>
        <p className="mt-3 text-sm leading-relaxed text-white/50">
          Traffic is split by Thompson sampling, not a fixed ratio: each arm is served with the
          probability that it is the best one, so losing arms fade out on their own. Reward is a
          track kept past the skip threshold.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          ["Impressions", report.total_impressions.toLocaleString()],
          ["Rewards", report.total_rewards.toLocaleString()],
          ["Overall rate", noData ? "—" : pct(rate)],
          ["Leading arm", noData ? "—" : leader.variant],
        ].map(([label, value]) => (
          <div key={label} className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
            <div className="text-[10px] uppercase tracking-wider text-white/35">{label}</div>
            <div className="mt-1 truncate font-mono text-lg tabular-nums">{value}</div>
          </div>
        ))}
      </div>

      {noData && (
        <p className="rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3 text-sm text-white/45">
          No traffic yet — every arm sits at its Beta(1,1) prior, which is why they all read 50%.
          Play and skip a few tracks and the posteriors will separate.
        </p>
      )}

      <div className="flex flex-col gap-3">
        {report.arms.map((arm) => {
          const leading = !noData && arm.variant === leader.variant;
          return (
            <div
              key={arm.variant}
              className={`rounded-xl border p-4 transition ${
                leading ? "border-emerald-400/25 bg-emerald-400/[0.04]" : "border-white/10"
              }`}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-sm">{arm.variant}</span>
                  {leading && (
                    <span className="rounded-full bg-emerald-400/15 px-2 py-0.5 text-[10px] text-emerald-300">
                      leading
                    </span>
                  )}
                </div>
                <span className="text-xs text-white/40 tabular-nums">
                  {arm.rewards}/{arm.impressions} · P(best){" "}
                  <span className="text-white/80">
                    {arm.probability_best === null ? "—" : pct(arm.probability_best)}
                  </span>
                </span>
              </div>

              <p className="mt-1 text-xs text-white/35">
                {ARM_DESCRIPTIONS[arm.variant] ?? "custom ranking weights"}
              </p>

              <div className="mt-3">
                <IntervalBar arm={arm} scale={scale} leading={leading} />
              </div>

              <div className="mt-1.5 text-[10px] text-white/30 tabular-nums">
                mean {pct(arm.posterior_mean)} · 95% CI [{pct(arm.ci_low)},{" "}
                {pct(arm.ci_high)}]
              </div>
            </div>
          );
        })}
      </div>

      <p className="max-w-2xl text-xs leading-relaxed text-white/30">
        P(best) is a Monte-Carlo estimate over all four posteriors at once — the quantity a ship
        decision should rest on. Comparing raw conversion rates would call a 5/10 arm a winner
        over a 400/1000 arm.
      </p>
    </div>
  );
}
