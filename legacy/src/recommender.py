import numpy as np
import pandas as pd


METADATA_COLS = [
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# 56D audio feature names — order must match extract_audio_features() in audio_feature_extractor.py
AUDIO_FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(1, 14)]
    + [f"mfcc{i}_std" for i in range(1, 14)]
    + ["chroma_C", "chroma_C#", "chroma_D", "chroma_D#", "chroma_E",
       "chroma_F", "chroma_F#", "chroma_G", "chroma_G#", "chroma_A",
       "chroma_A#", "chroma_B"]
    + ["centroid_mean", "centroid_std",
       "bandwidth_mean", "bandwidth_std",
       "rolloff_mean", "rolloff_std",
       "zcr_mean", "zcr_std",
       "rms_mean", "rms_std"]
    + ["audio_tempo"]
    + [f"spec_contrast_{i}" for i in range(1, 8)]
)

# 64D full feature space used by K-Means (meta + audio)
FULL_FEATURE_NAMES = METADATA_COLS + AUDIO_FEATURE_NAMES


def build_embedding_std(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (emb_std, feat_mean, feat_std) where emb_std is the embedding matrix
    z-score standardized per feature dimension using the training distribution.

    Use this — not the L2-normalized emb_matrix — when matching a new song that
    has no Spotify metadata. Unlike cosine similarity (which is magnitude-invariant),
    Euclidean distance on z-scored features respects absolute feature magnitudes:
    [0.1, 0.1, ...] and [0.9, 0.9, ...] correctly produce a non-zero distance
    instead of being treated as identical unit vectors.
    """
    raw = np.stack(df['embedding'].values).astype(np.float32)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (raw - mean) / std, mean, std


def build_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized metadata and embedding matrices from master dataframe."""
    meta = df[METADATA_COLS].copy()
    # Normalize tempo to [0, 1] so MSE is comparable across features
    meta['tempo'] = meta['tempo'] / 200.0
    meta_matrix = meta.values.astype(np.float32)

    emb_matrix = np.stack(df['embedding'].values).astype(np.float32)

    # 1. Z-score Standardize per feature dimension to prevent Cosine Collapse
    # This ensures large features (like MFCC_0) do not dominate the Euclidean shape.
    feat_mean = emb_matrix.mean(axis=0)
    feat_std = emb_matrix.std(axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)
    emb_matrix = (emb_matrix - feat_mean) / feat_std

    # 2. L2-normalize each embedding row so cosine similarity = dot product
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms

    return meta_matrix, emb_matrix


def cosine_similarity(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row of A and vector b.
    Cosine similarity measures the angle between vectors, not magnitude,
    making it robust in high-dimensional spaces where MSE suffers from
    the curse of dimensionality.
    """
    # L2-normalize each row of A and vector b
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_norms = np.where(A_norms == 0, 1.0, A_norms)
    b_norm = np.linalg.norm(b)
    b_norm = b_norm if b_norm != 0 else 1.0

    # Dot product of normalized vectors = cosine similarity
    return (A / A_norms) @ (b / b_norm)


def recommend(
    query_idx: int,
    df: pd.DataFrame,
    meta_matrix: np.ndarray,
    emb_matrix: np.ndarray,
    lambda_weight: float = 0.5,
    top_k: int = 5,
    genre_filter: str = None,
) -> list[dict]:
    """
    Recommend songs using a hybrid multimodal distance metric.
    
    Formula: score = MSE(metadata) + λ * (1 - CosineSimilarity(audio))
    
    Mathematical Rationale:
    1. MSE on Metadata (8D):
       Metadata features (e.g., energy, valence) are bounded [0, 1]. In such low-dimensional 
       spaces, Euclidean/MSE distance is a valid metric for absolute "vibe" differences.
       
    2. Cosine on Audio (56D):
       High-dimensional vectors suffer from the "Curse of Dimensionality" where Euclidean 
       distances tend to concentrate (making songs appear equally distant). Cosine Similarity 
       circumvents this by focusing on the *angle* (semantic direction) between the 
       acoustic signatures rather than their magnitude.
       
    3. λ (Lambda):
       Balances the two distance systems. A tuned λ=0.1 ensures that metadata acts as 
       the primary filter while audio embeddings provide fine-grained timbral matching.
    """
    q_meta = meta_matrix[query_idx]
    q_emb = emb_matrix[query_idx]

    # --- Step 1: Metadata Distance (MSE) ---
    # We measure absolute deviation in track characteristics. 
    # Small differences (e.g., 0.1 vs 0.12 tempo) yield near-zero scores.
    meta_dists = np.mean((meta_matrix - q_meta) ** 2, axis=1)

    # --- Step 2: Audio Embedding Distance (1 - Cosine) ---
    # We focus on the "Acoustic Signature" direction.
    # Note: emb_matrix was Z-scored and L2-normalized during build_matrices, 
    # so (1 - dot_product) is equivalent to 0.5 * squared_euclidean(normalized_vectors).
    cos_sim = cosine_similarity(emb_matrix, q_emb)
    emb_dists = 1.0 - cos_sim

    # --- Step 3: Hybrid Fusion ---
    scores = meta_dists + lambda_weight * emb_dists

    if genre_filter and genre_filter != "All":
        mask = df['playlist_genre'] != genre_filter
        scores[mask] = np.inf

    scores[query_idx] = np.inf

    top_indices = np.argsort(scores)[:top_k]

    return [
        {
            'idx': int(idx),
            'score': float(scores[idx]),
            'meta_dist': float(meta_dists[idx]),
            'emb_dist': float(emb_dists[idx]),
        }
        for idx in top_indices
    ]


def sanity_check(
    song_a: str,
    song_b: str,
    song_c: str,
    df: pd.DataFrame,
    meta_matrix: np.ndarray,
    emb_matrix: np.ndarray,
    lambda_weight: float = 0.5,
) -> dict:
    """
    Verify that song_a is closer to song_b than to song_c.
    Used to evaluate whether the distance metric makes musical sense.
    """
    def find_idx(name):
        matches = df[df['track_name'].str.lower() == name.lower()]
        return matches.index[0] if not matches.empty else None

    idx_a, idx_b, idx_c = find_idx(song_a), find_idx(song_b), find_idx(song_c)
    if any(i is None for i in [idx_a, idx_b, idx_c]):
        return {"error": "One or more songs not found in dataset."}

    def score(i, j):
        m = float(np.mean((meta_matrix[i] - meta_matrix[j]) ** 2))
        cos = float(cosine_similarity(emb_matrix[i:i+1], emb_matrix[j])[0])
        e = 1.0 - cos
        return m + lambda_weight * e, m, e

    score_ab, m_ab, e_ab = score(idx_a, idx_b)
    score_ac, m_ac, e_ac = score(idx_a, idx_c)

    return {
        'song_a': song_a, 'song_b': song_b, 'song_c': song_c,
        'score_ab': score_ab, 'score_ac': score_ac,
        'meta_ab': m_ab, 'meta_ac': m_ac,
        'emb_ab': e_ab, 'emb_ac': e_ac,
        'passed': score_ab < score_ac,
    }
