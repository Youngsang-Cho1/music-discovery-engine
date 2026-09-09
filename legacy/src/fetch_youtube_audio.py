import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import yt_dlp

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from src.audio_utils import get_safe_name
except ImportError:
    from audio_utils import get_safe_name

BATCH_SIZE = 1000
SLEEP_BETWEEN_MIN = 5.0
SLEEP_BETWEEN_MAX = 10.0
SLEEP_ON_ERROR = 15.0
OUTPUT_DIR = PROJECT_ROOT / "data/audio_files"
MANIFEST_PATH = PROJECT_ROOT / "data/dataset/download_manifest.csv"
FAIL_LOG_PATH = PROJECT_ROOT / "data/dataset/download_failures.jsonl"


def build_or_load_manifest(full_csv: Path, existing_csv: Path) -> pd.DataFrame:
    """Create a stable shuffled manifest once, then reuse it across batches."""
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)

    df_full = pd.read_csv(full_csv)
    existing_ids = set(pd.read_csv(existing_csv)["track_id"].tolist())

    remaining = (
        df_full[~df_full["track_id"].isin(existing_ids)]
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
        .copy()
    )
    remaining["safe_name"] = remaining.apply(
        lambda r: get_safe_name(r["track_name"], r["track_artist"]), axis=1
    )
    remaining.to_csv(MANIFEST_PATH, index=False)
    return remaining


def already_downloaded(manifest: pd.DataFrame, processed_csv: Path) -> set[str]:
    """Return safe_names already in processed_songs.csv or present as files or permanently failed."""
    done = set()

    # Check processed_songs.csv via manifest mapping
    if processed_csv.exists():
        processed_ids = set(pd.read_csv(processed_csv)["track_id"].tolist())
        done |= set(manifest.loc[manifest["track_id"].isin(processed_ids), "safe_name"].tolist())

    # Also check actual files (current batch in progress)
    if OUTPUT_DIR.exists():
        for p in OUTPUT_DIR.iterdir():
            if p.is_file():
                done.add(p.stem)

    # Skip previously failed tracks (video unavailable, not found, etc.)
    if FAIL_LOG_PATH.exists():
        try:
            with open(FAIL_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        safe_name = get_safe_name(entry["track_name"], entry["artist_name"])
                        done.add(safe_name)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

    return done


def log_failure(track_name: str, artist_name: str, query: str, error: str) -> None:
    FAIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAIL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "track_name": track_name,
                    "artist_name": artist_name,
                    "query": query,
                    "error": error,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def find_downloaded_file(video_id: str) -> Path | None:
    """Find the actual file yt-dlp/ffmpeg produced for a given video_id."""
    candidates = list(OUTPUT_DIR.glob(f"{video_id}.*"))
    if not candidates:
        return None

    preferred_suffixes = [".m4a", ".mp3", ".webm", ".opus"]
    candidates.sort(key=lambda p: preferred_suffixes.index(p.suffix) if p.suffix in preferred_suffixes else 999)
    return candidates[0]


def download_one(ydl: yt_dlp.YoutubeDL, track_name: str, artist_name: str) -> bool:
    safe_name = get_safe_name(track_name, artist_name)

    for existing in OUTPUT_DIR.glob(f"{safe_name}.*"):
        if existing.is_file():
            return True

    query = f"ytsearch1:{track_name} {artist_name} audio"
    try:
        info = ydl.extract_info(query, download=True)
        entries = info.get("entries", [])
        if not entries:
            log_failure(track_name, artist_name, query, "No search entries returned")
            return False

        video_id = entries[0]["id"]
        downloaded = find_downloaded_file(video_id)
        if downloaded is None:
            log_failure(track_name, artist_name, query, f"Downloaded file for video_id={video_id} not found")
            return False

        final_path = OUTPUT_DIR / f"{safe_name}{downloaded.suffix}"

        if final_path.exists():
            downloaded.unlink(missing_ok=True)
        else:
            downloaded.rename(final_path)

        return True

    except Exception as e:
        error_msg = repr(e).lower()
        
        # Check if it's a transient error (rate limit, bot ban, network failure)
        is_transient = False
        transient_keywords = [
            "bot", 
            "429", 
            "too many requests", 
            "rate-limit", 
            "ssl", 
            "timed out", 
            "connection reset", 
            "unavailable"
        ]
        
        for k in transient_keywords:
            if k in error_msg:
                is_transient = True
                break
                
        # Handle "Sign in to confirm" carefully: "bot" is transient, "age" is permanent
        if "sign in to confirm" in error_msg and "age" not in error_msg:
            is_transient = True
            
        if not is_transient:
            # Only permanently log legitimate failures (Age restricted, video not found, etc.)
            log_failure(track_name, artist_name, query, repr(e))
            
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch YouTube audio downloader")
    parser.add_argument("--batch", type=int, required=True, help="Batch index (0-based, 1000 songs each)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_csv = PROJECT_ROOT / "data/dataset/spotify_songs_full.csv"
    existing_csv = PROJECT_ROOT / "data/dataset/processed_songs.csv"

    manifest = build_or_load_manifest(full_csv, existing_csv)

    start = args.batch * BATCH_SIZE
    batch = manifest.iloc[start:start + BATCH_SIZE].copy()

    if batch.empty:
        print(f"Batch {args.batch}: nothing left to download.")
        return

    done = already_downloaded(manifest, existing_csv)
    todo = batch[~batch["safe_name"].isin(done)].copy()

    print(f"Batch {args.batch}: {len(batch)} songs | {len(todo)} to download | {len(batch) - len(todo)} already done")
    print(f"Output: {OUTPUT_DIR}\n")

    success, failed = 0, 0

    ydl_opts = {
        "format": "bestaudio[ext=m4a]",
        "outtmpl": str(OUTPUT_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            ok = download_one(ydl, row["track_name"], row["track_artist"])
            print(f"[{i:4d}/{len(todo)}] {'✓' if ok else '✗'}  {row['track_name']} — {row['track_artist']}")

            if ok:
                success += 1
                time.sleep(random.uniform(SLEEP_BETWEEN_MIN, SLEEP_BETWEEN_MAX))
            else:
                failed += 1
                time.sleep(SLEEP_ON_ERROR)

    print(f"\nBatch {args.batch} done: {success} success / {failed} failed")
    print(f"Total audio files now: {len([p for p in OUTPUT_DIR.iterdir() if p.is_file()])}")
    print("\nNext steps:")
    print("  python src/audio_feature_extractor.py")
    print("  python src/pca.py")
    print("  python src/clustering.py")
    print("  python src/build_master_dataset.py")


if __name__ == "__main__":
    main()
