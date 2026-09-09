export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-white/[0.06] ${className}`} />;
}

/** Row-shaped placeholder, so the list doesn't jump when results arrive. */
export function TrackRowSkeleton() {
  return (
    <div className="flex items-center gap-3 px-2 py-2">
      <Skeleton className="h-3 w-6" />
      <Skeleton className="size-11 shrink-0" />
      <div className="flex flex-1 flex-col gap-1.5">
        <Skeleton className="h-3 w-1/3" />
        <Skeleton className="h-2.5 w-1/4" />
      </div>
      <Skeleton className="h-3 w-10" />
    </div>
  );
}

export function ResultsSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-1" aria-busy="true" aria-label="Loading results">
      {Array.from({ length: rows }, (_, i) => (
        <TrackRowSkeleton key={i} />
      ))}
    </div>
  );
}
