"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { logEvent } from "@/lib/api";
import { getSessionId } from "@/lib/session";
import type { Track } from "@/lib/types";

/** Which ranking produced the track being played — needed for attribution. */
export interface PlayContext {
  variant: string;
  seedTrackId?: string | null;
  position?: number | null;
}

interface QueueItem {
  track: Track;
  context: PlayContext;
}

interface PlayerState {
  current: Track | null;
  isPlaying: boolean;
  progress: number;
  duration: number;
  play: (track: Track, context: PlayContext, queue?: QueueItem[]) => void;
  toggle: () => void;
  next: () => void;
  seek: (seconds: number) => void;
  like: () => void;
}

const PlayerContext = createContext<PlayerState | null>(null);

export function usePlayer(): PlayerState {
  const ctx = useContext(PlayerContext);
  if (!ctx) throw new Error("usePlayer must be used inside <PlayerProvider>");
  return ctx;
}

export function PlayerProvider({ children }: { children: React.ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [current, setCurrent] = useState<Track | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  // Playback bookkeeping lives in refs, not state: it is read inside event
  // handlers that must not re-subscribe on every render.
  const contextRef = useRef<PlayContext | null>(null);
  const startedAtRef = useRef<number>(0);
  const listenedMsRef = useRef<number>(0);
  const resolvedRef = useRef<boolean>(false);
  const queueRef = useRef<QueueItem[]>([]);

  // Mirror `current` into a ref so unload/ended handlers can read the live
  // value without being re-registered on every track change.
  const currentRef = useRef<Track | null>(null);
  useEffect(() => {
    currentRef.current = current;
  }, [current]);

  const emit = useCallback(
    (
      action: "play" | "skip" | "complete" | "like",
      track: Track,
      context: PlayContext,
      dwellMs: number | null,
      beacon = false,
    ) => {
      logEvent(
        {
          session_id: getSessionId(),
          variant: context.variant,
          track_id: track.id,
          seed_track_id: context.seedTrackId ?? null,
          position: context.position ?? null,
          action,
          dwell_ms: dwellMs,
        },
        { beacon },
      );
    },
    [],
  );

  /**
   * Close out the currently playing track.
   *
   * Every impression must resolve exactly once, into either a completion or a
   * skip — resolving twice would double-count the outcome into the arm's
   * posterior, and not resolving at all leaves the arm's denominator wrong.
   */
  const resolveCurrent = useCallback(
    (outcome: "skip" | "complete", beacon = false) => {
      const track = currentRef.current;
      const context = contextRef.current;
      if (!track || !context || resolvedRef.current) return;

      resolvedRef.current = true;
      const dwell = listenedMsRef.current + (Date.now() - startedAtRef.current);
      emit(outcome, track, context, Math.max(0, Math.round(dwell)), beacon);
    },
    [emit],
  );

  const play = useCallback(
    (track: Track, context: PlayContext, queue: QueueItem[] = []) => {
      if (!track.audio_url) return;

      // Starting a new track abandons the previous one.
      resolveCurrent("skip");

      queueRef.current = queue;
      contextRef.current = context;
      startedAtRef.current = Date.now();
      listenedMsRef.current = 0;
      resolvedRef.current = false;

      setCurrent(track);
      setProgress(0);

      const audio = audioRef.current;
      if (audio) {
        audio.src = track.audio_url;
        void audio.play().catch(() => setIsPlaying(false));
      }
      emit("play", track, context, null);
    },
    [emit, resolveCurrent],
  );

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || !current) return;

    if (audio.paused) {
      startedAtRef.current = Date.now();
      void audio.play().catch(() => setIsPlaying(false));
    } else {
      // Bank the elapsed time so a pause doesn't inflate dwell.
      listenedMsRef.current += Date.now() - startedAtRef.current;
      audio.pause();
    }
  }, [current]);

  const next = useCallback(() => {
    const queue = queueRef.current;
    const currentId = currentRef.current?.id;
    const idx = queue.findIndex((item) => item.track.id === currentId);
    const upcoming = queue[idx + 1];
    if (upcoming) play(upcoming.track, upcoming.context, queue);
  }, [play]);

  const seek = useCallback((seconds: number) => {
    const audio = audioRef.current;
    if (audio) audio.currentTime = seconds;
  }, []);

  const like = useCallback(() => {
    const track = currentRef.current;
    const context = contextRef.current;
    if (!track || !context) return;
    const dwell = listenedMsRef.current + (Date.now() - startedAtRef.current);
    emit("like", track, context, Math.max(0, Math.round(dwell)));
  }, [emit]);

  // Audio element wiring.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTime = () => setProgress(audio.currentTime);
    const onMeta = () => setDuration(audio.duration || 0);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => {
      resolveCurrent("complete");
      setIsPlaying(false);
      next();
    };

    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("loadedmetadata", onMeta);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);

    return () => {
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("loadedmetadata", onMeta);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
    };
  }, [next, resolveCurrent]);

  // Closing the tab mid-track is a skip. sendBeacon survives unload; fetch
  // does not, and losing these would bias every arm upward.
  useEffect(() => {
    const onHide = () => {
      if (document.visibilityState === "hidden") resolveCurrent("skip", true);
    };
    document.addEventListener("visibilitychange", onHide);
    return () => document.removeEventListener("visibilitychange", onHide);
  }, [resolveCurrent]);

  const value = useMemo(
    () => ({ current, isPlaying, progress, duration, play, toggle, next, seek, like }),
    [current, isPlaying, progress, duration, play, toggle, next, seek, like],
  );

  return (
    <PlayerContext.Provider value={value}>
      {children}
      <audio ref={audioRef} preload="metadata" />
    </PlayerContext.Provider>
  );
}
