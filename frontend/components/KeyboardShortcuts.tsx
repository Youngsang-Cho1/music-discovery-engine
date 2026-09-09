"use client";

import { useEffect } from "react";

import { usePlayer } from "./PlayerProvider";

/**
 * Global player shortcuts.
 *
 * Deliberately skipped while a text input has focus — Space is a character
 * there, and hijacking it would make the search box unusable.
 */
export function KeyboardShortcuts() {
  const { current, toggle, next, seek, progress, duration, like } = usePlayer();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target?.isContentEditable;
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (!current) return;

      switch (e.key) {
        case " ":
          e.preventDefault(); // otherwise the page scrolls
          toggle();
          break;
        case "ArrowRight":
          e.preventDefault();
          seek(Math.min(progress + 10, duration || progress));
          break;
        case "ArrowLeft":
          e.preventDefault();
          seek(Math.max(progress - 10, 0));
          break;
        case "n":
          next();
          break;
        case "l":
          like();
          break;
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, toggle, next, seek, progress, duration, like]);

  return null;
}
