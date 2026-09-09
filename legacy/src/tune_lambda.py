"""
Evaluation + λ tuning for the recommendation score:
    score = MSE_metadata + λ × (1 − cos_sim_embedding)

Produces three outputs:
  1. λ sweep — pick the best λ based on same-artist vs random-pair discrimination
  2. Ablation study — compare hybrid to each modality alone, showing both are needed
  3. Precision@5 — for 100 random seed songs whose artist has ≥2 tracks in the DB,
     measure what fraction of the top-5 recommendations share the seed's artist.
     Higher = the metric actually surfaces the "right" neighbors.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import build_matrices, cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent
N_PAIRS = 100
N_SEEDS = 100
K = 5
RANDOM_SEED = 42


def build_same_artist_pairs(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    artist_groups = df.groupby('track_artist').indices
    valid_artists = [a for a, idxs in artist_groups.items() if len(idxs) >= 2]
    pairs = []
    for _ in range(n):
        artist = rng.choice(valid_artists)
        idxs = artist_groups[artist]
        i, j = rng.choice(idxs, size=2, replace=False)
        pairs.append((int(i), int(j)))
    return pairs


def build_random_pairs(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    pairs = []
    while len(pairs) < n:
        i, j = rng.integers(0, len(df), size=2)
        if df.iloc[i]['track_artist'] != df.iloc[j]['track_artist']:
            pairs.append((int(i), int(j)))
    return pairs


def pair_score(meta: np.ndarray, emb: np.ndarray, i: int, j: int, mode: str, lam: float) -> float:
    """Distance between two songs under the chosen scoring mode."""
    if mode == 'mse_only':
        return float(np.mean((meta[i] - meta[j]) ** 2))
    if mode == 'cos_only':
        cos = float(cosine_similarity(emb[i:i+1], emb[j])[0])
        return 1.0 - cos
    m = float(np.mean((meta[i] - meta[j]) ** 2))
    cos = float(cosine_similarity(emb[i:i+1], emb[j])[0])
    return m + lam * (1.0 - cos)


def pass_rate(meta, emb, same_pairs, rand_pairs, mode: str, lam: float) -> float:
    same = np.array([pair_score(meta, emb, i, j, mode, lam) for i, j in same_pairs])
    rand = np.array([pair_score(meta, emb, i, j, mode, lam) for i, j in rand_pairs])
    return float(np.mean(same[:, None] < rand[None, :]))


def precision_at_k(df: pd.DataFrame, meta: np.ndarray, emb: np.ndarray,
                   mode: str, lam: float, k: int, n_seeds: int,
                   rng: np.random.Generator) -> float:
    """For each seed, fraction of top-k neighbors sharing the seed's artist. Averaged across seeds."""
    artist_counts = df['track_artist'].value_counts()
    multi_artist = set(artist_counts[artist_counts >= 2].index)
    eligible = df[df['track_artist'].isin(multi_artist)].index.tolist()
    n_seeds = min(n_seeds, len(eligible))

    seeds = rng.choice(eligible, size=n_seeds, replace=False)

    # Precompute embedding norms (needed only for cos-based modes)
    emb_norms = np.linalg.norm(emb, axis=1)
    emb_norms = np.where(emb_norms == 0, 1.0, emb_norms)

    total_hits = 0
    for seed_idx in seeds:
        seed_artist = df.iloc[seed_idx]['track_artist']

        if mode == 'mse_only':
            scores = np.mean((meta - meta[seed_idx]) ** 2, axis=1)
        elif mode == 'cos_only':
            q_norm = np.linalg.norm(emb[seed_idx]) or 1.0
            cos = (emb @ emb[seed_idx]) / (emb_norms * q_norm)
            scores = 1.0 - cos
        else:
            meta_d = np.mean((meta - meta[seed_idx]) ** 2, axis=1)
            q_norm = np.linalg.norm(emb[seed_idx]) or 1.0
            cos = (emb @ emb[seed_idx]) / (emb_norms * q_norm)
            scores = meta_d + lam * (1.0 - cos)

        scores[seed_idx] = np.inf
        top_k = np.argsort(scores)[:k]
        hits = sum(1 for idx in top_k if df.iloc[idx]['track_artist'] == seed_artist)
        total_hits += hits

    return total_hits / (n_seeds * k)


def main():
    data_path = PROJECT_ROOT / 'data/dataset/master_music_data.parquet'
    if not data_path.exists():
        print(f"Missing {data_path}. Run build_master_dataset.py first.")
        return

    df = pd.read_parquet(data_path)
    meta, emb = build_matrices(df)
    rng = np.random.default_rng(RANDOM_SEED)

    same_pairs = build_same_artist_pairs(df, N_PAIRS, rng)
    rand_pairs = build_random_pairs(df, N_PAIRS, rng)

    # ── (1) λ sweep ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("(1) λ Sweep — same-artist vs random-pair discrimination")
    print("=" * 70)
    print(f"{'λ':>6} | {'same_avg':>10} | {'rand_avg':>10} | {'pass_rate':>10}")
    print("-" * 55)

    lambdas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    results = []
    for lam in lambdas:
        same_scores = np.array([pair_score(meta, emb, i, j, 'hybrid', lam) for i, j in same_pairs])
        rand_scores = np.array([pair_score(meta, emb, i, j, 'hybrid', lam) for i, j in rand_pairs])
        pr = float(np.mean(same_scores[:, None] < rand_scores[None, :]))
        results.append({'lambda': lam, 'pass_rate': pr})
        print(f"{lam:>6.2f} | {same_scores.mean():>10.6f} | {rand_scores.mean():>10.6f} | {pr:>10.4f}")

    best_lam = max(results, key=lambda r: r['pass_rate'])['lambda']
    print(f"\nBest λ: {best_lam} (random-chance baseline = 0.5)")

    # ── (2) Ablation: hybrid vs each modality alone ──────────────────────────
    print("\n" + "=" * 70)
    print(f"(2) Ablation Study — Pass rate at the chosen λ={best_lam}")
    print("=" * 70)
    print(f"{'Variant':<25} | {'pass_rate':>10}")
    print("-" * 45)
    pr_mse = pass_rate(meta, emb, same_pairs, rand_pairs, 'mse_only', 0.0)
    pr_cos = pass_rate(meta, emb, same_pairs, rand_pairs, 'cos_only', 0.0)
    pr_hybrid = pass_rate(meta, emb, same_pairs, rand_pairs, 'hybrid', best_lam)
    print(f"{'MSE only (metadata)':<25} | {pr_mse:>10.4f}")
    print(f"{'Cosine only (audio)':<25} | {pr_cos:>10.4f}")
    print(f"{f'Hybrid (λ={best_lam})':<25} | {pr_hybrid:>10.4f}")

    # ── (3) Precision@K ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"(3) Precision@{K} — fraction of top-{K} sharing seed's artist ({N_SEEDS} seeds)")
    print("=" * 70)
    print(f"{'Variant':<25} | {f'P@{K}':>10}")
    print("-" * 45)
    rng = np.random.default_rng(RANDOM_SEED)  # reset for reproducibility
    p_mse = precision_at_k(df, meta, emb, 'mse_only', 0.0, K, N_SEEDS, rng)
    rng = np.random.default_rng(RANDOM_SEED)
    p_cos = precision_at_k(df, meta, emb, 'cos_only', 0.0, K, N_SEEDS, rng)
    rng = np.random.default_rng(RANDOM_SEED)
    p_hybrid = precision_at_k(df, meta, emb, 'hybrid', best_lam, K, N_SEEDS, rng)
    print(f"{'MSE only (metadata)':<25} | {p_mse:>10.4f}")
    print(f"{'Cosine only (audio)':<25} | {p_cos:>10.4f}")
    print(f"{f'Hybrid (λ={best_lam})':<25} | {p_hybrid:>10.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Chosen λ           : {best_lam}")
    print(f"Pass rate (hybrid) : {pr_hybrid:.4f}  (random baseline 0.5)")
    print(f"Precision@{K}       : {p_hybrid:.4f}  (random baseline ≈ {1/len(df):.4f})")
    print(f"Hybrid beats MSE-only by     : {(pr_hybrid - pr_mse) * 100:+.2f} pp (pass_rate)")
    print(f"Hybrid beats cosine-only by  : {(pr_hybrid - pr_cos) * 100:+.2f} pp (pass_rate)")


if __name__ == "__main__":
    main()