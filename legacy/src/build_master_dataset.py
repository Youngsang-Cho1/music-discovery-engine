import pandas as pd
import numpy as np
from pathlib import Path

try:
    from src.audio_utils import get_safe_name
except ImportError:
    from audio_utils import get_safe_name

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    """
    Fuses textual metadata with high-dimensional audio embeddings.
    
    This process is critical for Multimodal Retrieval:
    1. It aligns Spotify's subjective features (danceability) with Librosa's objective features (MFCC).
    2. It uses 'safe_name' (Artist_Title slug) as a robust join key since raw track_ids 
       might differ between the CSV and the YouTube crawler results.
    3. It enforces data integrity by dropping duplicates, ensuring every vector in the
       master dataset corresponds to a unique acoustic fingerprint.
    """
    processed_csv = PROJECT_ROOT / 'data/dataset/processed_songs.csv'
    embedding_parquet = PROJECT_ROOT / 'data/embeddings/audio_features.parquet'
    output_master = PROJECT_ROOT / 'data/dataset/master_music_data.parquet'

    print("Building Master Dataset...")

    if not processed_csv.exists():
        print(f"Error: Missing {processed_csv}")
        return
    if not embedding_parquet.exists():
        print(f"Error: Missing {embedding_parquet}")
        return

    # Load both modalities
    df_meta = pd.read_csv(processed_csv)
    df_emb = pd.read_parquet(embedding_parquet)

    # Sanitize keys for joining
    df_meta['safe_name'] = df_meta.apply(
        lambda r: get_safe_name(r['track_name'], r['track_artist']), axis=1
    )
    df_emb = df_emb.rename(columns={'track_id': 'safe_name'})

    # Inner join captures only songs for which we have both Meta and Audio
    master_df = pd.merge(df_meta, df_emb, on='safe_name', how='inner')
    
    # Drop duplicates to prevent 'Multiple Candidates' errors during retrieval
    master_df = master_df.drop_duplicates(subset=['safe_name']).reset_index(drop=True)

    if master_df.empty:
        print("Error: Join produced 0 rows. Check safe_name keys.")
        return

    print(f"Master Dataset: {len(master_df)} songs, {master_df.shape[1]} columns.")
    master_df.to_parquet(output_master)
    print(f"Saved to {output_master}")


if __name__ == "__main__":
    main()
