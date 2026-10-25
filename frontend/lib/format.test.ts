import { describe, expect, it } from "vitest";

import {
  dominantSignal,
  formatDuration,
  formatPlays,
  readableFeatures,
  reachTier,
} from "./format";
import type { Signals } from "./types";

const signals = (partial: Partial<Signals>): Signals => ({
  audio: 0,
  descriptor: 0,
  tag: 0,
  novelty: 0,
  ...partial,
});

describe("formatDuration", () => {
  it("formats minutes and seconds", () => {
    expect(formatDuration(125)).toBe("2:05");
    expect(formatDuration(59)).toBe("0:59");
    expect(formatDuration(3600)).toBe("60:00");
  });

  it("renders a placeholder rather than NaN for missing values", () => {
    expect(formatDuration(null)).toBe("--:--");
    expect(formatDuration(undefined)).toBe("--:--");
    expect(formatDuration(0)).toBe("--:--");
  });
});

describe("dominantSignal", () => {
  it("names the strongest contribution", () => {
    expect(dominantSignal(signals({ audio: 0.9, tag: 0.2 }))).toBe("similar sound");
    expect(dominantSignal(signals({ novelty: 0.8, audio: 0.3 }))).toBe("hidden gem");
    expect(dominantSignal(signals({ tag: 0.6, descriptor: 0.5 }))).toBe("shared genre");
  });

  it("says nothing when no signal contributed", () => {
    // A row claiming a reason that is all zeros would be an outright lie.
    expect(dominantSignal(signals({}))).toBeNull();
    expect(dominantSignal(undefined)).toBeNull();
  });
});

describe("readableFeatures", () => {
  it("keeps tempo in its real unit", () => {
    const [tempo] = readableFeatures({ tempo_bpm: 128 });
    expect(tempo.label).toBe("Tempo");
    expect(tempo.value).toBe("128 BPM");
  });

  it("translates unitless descriptors into words", () => {
    const quiet = readableFeatures({ loudness_db: -55 })[0];
    const loud = readableFeatures({ loudness_db: -3 })[0];
    expect(quiet.value).toBe("quiet");
    expect(loud.value).toBe("loud");
  });

  it("maps each descriptor onto 0-1 for bar widths", () => {
    const [tempo] = readableFeatures({ tempo_bpm: 130 });
    // (130 - 40) / (220 - 40) = 0.5
    expect(tempo.fraction).toBeCloseTo(0.5, 2);
  });

  it("clamps values outside the conventional range", () => {
    // Beat tracking octave errors produce implausible BPMs; a fraction above 1
    // would render a bar wider than its container.
    expect(readableFeatures({ tempo_bpm: 9000 })[0].fraction).toBe(1);
    expect(readableFeatures({ loudness_db: -999 })[0].fraction).toBe(0);
  });

  it("orders output independently of key order", () => {
    const a = readableFeatures({ tempo_bpm: 120, loudness_db: -10 }).map((f) => f.key);
    const b = readableFeatures({ loudness_db: -10, tempo_bpm: 120 }).map((f) => f.key);
    expect(a).toEqual(b);
  });

  it("ignores descriptors it has no wording for", () => {
    expect(readableFeatures({ unknown_metric: 5 })).toEqual([]);
  });

  it("handles an empty feature set", () => {
    expect(readableFeatures({})).toEqual([]);
  });
});

describe("formatPlays", () => {
  it("leaves counts under a thousand exact", () => {
    expect(formatPlays(0)).toBe("0");
    expect(formatPlays(642)).toBe("642");
    expect(formatPlays(999)).toBe("999");
  });

  it("switches to K and M at the right boundaries", () => {
    expect(formatPlays(1_000)).toBe("1.0K");
    expect(formatPlays(9_949)).toBe("9.9K");
    expect(formatPlays(214_647)).toBe("215K");
    expect(formatPlays(1_672_904)).toBe("1.7M");
    expect(formatPlays(12_400_000)).toBe("12M");
  });

  it("treats missing and nonsense counts as zero rather than throwing", () => {
    expect(formatPlays(null)).toBe("0");
    expect(formatPlays(undefined)).toBe("0");
    expect(formatPlays(Number.NaN)).toBe("0");
    expect(formatPlays(-5)).toBe("0");
  });
});

describe("reachTier", () => {
  it("separates unheard from merely obscure", () => {
    // The boundary the discovery arm is judged on: 0 is a different claim
    // from 1, not a smaller version of it.
    expect(reachTier(0)).toBe("unheard");
    expect(reachTier(1)).toBe("deep-cut");
    expect(reachTier(999)).toBe("deep-cut");
    expect(reachTier(1_000)).toBe("popular");
  });

  it("treats a missing count as unheard", () => {
    expect(reachTier(null)).toBe("unheard");
    expect(reachTier(undefined)).toBe("unheard");
  });
});
