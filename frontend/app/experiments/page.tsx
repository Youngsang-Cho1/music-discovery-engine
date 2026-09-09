"use client";

import { useEffect, useState } from "react";

import { getExperimentReport } from "@/lib/api";
import type { ExperimentReport } from "@/lib/types";

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
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
      <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
        {error}
      </p>
    );
  }
  if (!report) return <p className="text-sm text-white/40">Loading…</p>;

  const widest = Math.max(...report.arms.map((a) => a.ci_high), 0.01);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Live ranking experiment</h1>
        <p className="mt-2 max-w-2xl text-sm text-white/50">
          Traffic is split by Thompson sampling, not a fixed A/B ratio: each arm is served
          with the probability that it is the best one, so losing arms fade out on their own.
          Reward is a track kept past the skip threshold.
        </p>
      </div>

      <div className="flex gap-6 text-sm">
        <div>
          <div className="text-white/40">Impressions</div>
          <div className="text-lg tabular-nums">{report.total_impressions.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-white/40">Rewards</div>
          <div className="text-lg tabular-nums">{report.total_rewards.toLocaleString()}</div>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        {report.arms.map((arm) => (
          <div key={arm.variant} className="rounded-lg border border-white/10 p-4">
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-sm">{arm.variant}</span>
              <span className="text-xs text-white/40 tabular-nums">
                {arm.rewards}/{arm.impressions} · P(best){" "}
                <span className="text-white/70">
                  {arm.probability_best === null ? "—" : pct(arm.probability_best)}
                </span>
              </span>
            </div>

            {/* Credible interval, drawn to scale. A wide bar means the arm has
                not earned a verdict yet — the point of showing it. */}
            <div className="relative mt-3 h-2 rounded bg-white/5">
              <div
                className="absolute h-2 rounded bg-white/25"
                style={{
                  left: `${(arm.ci_low / widest) * 100}%`,
                  width: `${((arm.ci_high - arm.ci_low) / widest) * 100}%`,
                }}
              />
              <div
                className="absolute h-2 w-0.5 bg-white"
                style={{ left: `${(arm.posterior_mean / widest) * 100}%` }}
              />
            </div>

            <div className="mt-1 text-[10px] text-white/30 tabular-nums">
              mean {pct(arm.posterior_mean)} · 95% CI [{pct(arm.ci_low)}, {pct(arm.ci_high)}]
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
