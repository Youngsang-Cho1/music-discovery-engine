"use client";

import { useState } from "react";

interface Props {
  src: string | null;
  alt: string;
  size?: number;
  className?: string;
}

/**
 * Album art with a deterministic fallback.
 *
 * Jamendo artwork 404s often enough that a broken-image icon would be a
 * regular occurrence. The fallback derives its hue from the title, so a track
 * without art still gets a stable identity rather than a grey box that looks
 * like a loading failure.
 */
export function Artwork({ src, alt, size = 40, className = "" }: Props) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) {
    let hash = 0;
    for (let i = 0; i < alt.length; i++) hash = (hash * 31 + alt.charCodeAt(i)) | 0;
    const hue = Math.abs(hash) % 360;

    return (
      <div
        aria-hidden
        style={{
          width: size,
          height: size,
          background: `linear-gradient(135deg, oklch(0.45 0.11 ${hue}), oklch(0.28 0.07 ${(hue + 45) % 360}))`,
        }}
        className={`flex shrink-0 items-center justify-center rounded ${className}`}
      >
        <span
          className="font-medium text-white/70"
          style={{ fontSize: Math.max(10, size * 0.34) }}
        >
          {alt.trim().charAt(0).toUpperCase() || "♪"}
        </span>
      </div>
    );
  }

  // Plain <img>: artwork comes from a Jamendo CDN host that isn't known at
  // build time, so next/image would need it allow-listed and would proxy every
  // thumbnail through the server for no benefit.
  return (
    /* eslint-disable-next-line @next/next/no-img-element */
    <img
      src={src}
      alt={alt}
      width={size}
      height={size}
      loading="lazy"
      onError={() => setFailed(true)}
      className={`shrink-0 rounded bg-white/5 object-cover ${className}`}
      style={{ width: size, height: size }}
    />
  );
}
