import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extracts a 58-dimensional feature vector from audio using Librosa.
    
    Dimensions:
    - MFCC (26D): mean + std. Captures timbre (the 'texture' of the sound).
    - Chroma (12D): mean. Captures harmonic/musical key information.
    - Spectral Centroid/BW/Rolloff (6D): Captures 'brightness' and 'richness'.
    - Zero Crossing Rate/RMS (4D): Captures percussiveness and loudness.
    - Tempo (1D): Beats per minute (BPM).
    - Spectral Contrast (9D): foreground vs background energy separation.
    """
    # Load audio (mono, 22.05kHz) - limited to first 60s for consistency
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

    features = []

    # 1. MFCCs - The most common feature in speech/music recognition (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1).tolist())
    features.extend(np.std(mfcc, axis=1).tolist())

    # 2. Chroma - Maps energy into 12 semitones of the musical octave
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1).tolist())

    # 3. Spectral Features - Shape of the frequency distribution
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spectral_centroid)))
    features.append(float(np.std(spectral_centroid)))

    spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spectral_bw)))
    features.append(float(np.std(spectral_bw)))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(rolloff)))
    features.append(float(np.std(rolloff)))

    # 4. Temporal Features - Percussiveness and Loudness
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    # 5. Rhythm
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    features.append(float(tempo[0]))

    # 6. Spectral Contrast - Separation of frequency peaks and valleys
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1).tolist())

    return np.array(features, dtype=np.float32)


def main():
    input_dir = PROJECT_ROOT / "data/audio_files"
    embed_dir = PROJECT_ROOT / "data/embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Gather all audio files
    audio_files = (
        glob.glob(str(input_dir / "*.m4a")) +
        glob.glob(str(input_dir / "*.mp3"))
    )

    if not audio_files:
        print(f"No audio files found in '{input_dir}'. Please run fetch_youtube_audio.py first.")
        return

    # Skip files already in parquet
    out_path = embed_dir / "audio_features.parquet"
    existing_ids = set()
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        existing_ids = set(existing_df["track_id"].tolist())
        print(f"Existing parquet: {len(existing_ids)} tracks. Skipping these.")

    todo_files = [p for p in audio_files if os.path.splitext(os.path.basename(p))[0] not in existing_ids]
    print(f"Found {len(audio_files)} audio files total, {len(todo_files)} new to process.")

    if not todo_files:
        print("Nothing new to extract.")
        # Still update processed_songs.csv below so newly downloaded tracks are registered
        all_embeddings = []
        track_ids = list(existing_ids)
    else:
        all_embeddings = []
        track_ids = []

        for i, audio_path in enumerate(todo_files, 1):
            filename = os.path.basename(audio_path)
            track_id = os.path.splitext(filename)[0]

            try:
                print(f"[{i}/{len(todo_files)}] Extracting: {filename}")
                features = extract_audio_features(audio_path)
                all_embeddings.append(features.tolist())
                track_ids.append(track_id)
            except Exception as e:
                print(f"    -> SKIPPED {filename}: {e}")

        if all_embeddings:
            feature_dim = len(all_embeddings[0])
            print(f"\nExtracted {feature_dim}D features for {len(all_embeddings)} new songs.")

            new_df = pd.DataFrame({"track_id": track_ids, "embedding": all_embeddings})

            # Append to existing parquet
            if out_path.exists():
                existing_df = pd.read_parquet(out_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["track_id"])
            else:
                combined_df = new_df

            combined_df.to_parquet(out_path)
            print(f"Saved {len(combined_df)} total tracks to {out_path}")

            # For downstream CSV update, use all track_ids including existing ones
            track_ids = combined_df["track_id"].tolist()
        else:
            track_ids = list(existing_ids)

    # Update spotify_songs.csv with newly downloaded tracks
    existing_csv = PROJECT_ROOT / "data/dataset/processed_songs.csv"
    manifest_path = PROJECT_ROOT / "data/dataset/download_manifest.csv"

    if not manifest_path.exists():
        print("No manifest found, skipping processed_songs.csv update.")
        return

    existing_df = pd.read_csv(existing_csv) if existing_csv.exists() else pd.DataFrame()
    existing_ids = set(existing_df["track_id"].tolist()) if not existing_df.empty else set()

    manifest = pd.read_csv(manifest_path)
    full_csv = PROJECT_ROOT / "data/dataset/spotify_songs_full.csv"
    df_full = pd.read_csv(full_csv)

    # safe_name → spotify track_id 매핑 후 추가
    extracted_safe_names = set(track_ids)
    matched = manifest[manifest["safe_name"].isin(extracted_safe_names)]
    matched_spotify_ids = set(matched["track_id"].tolist())

    new_tracks = df_full[df_full["track_id"].isin(matched_spotify_ids) & ~df_full["track_id"].isin(existing_ids)]
    if not new_tracks.empty:
        updated_df = pd.concat([existing_df, new_tracks], ignore_index=True).drop_duplicates(subset=["track_id"])
        updated_df.to_csv(existing_csv, index=False)
        print(f"Updated {existing_csv} with {len(new_tracks)} new tracks.")
    else:
        print(f"No new tracks to add to {existing_csv}")


if __name__ == "__main__":
    main()

