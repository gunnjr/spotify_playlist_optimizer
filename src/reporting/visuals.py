from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sample_matrix(S: np.ndarray, df: pd.DataFrame, max_n: int = 80, seed: int = 42) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    n = len(df)
    if n <= max_n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
    return S[np.ix_(idx, idx)], df.iloc[idx].reset_index(drop=True), idx

def plot_similarity_heatmap(S: np.ndarray, labels: List[str] | None, out_png: str = "similarity_heatmap.png", title: str = "Similarity heatmap"):
    plt.figure(figsize=(8, 7))
    plt.imshow(S, interpolation="nearest", aspect="auto")
    plt.colorbar(label="similarity (0..1)")
    if labels is not None and len(labels) == S.shape[0] and len(labels) <= 50:
        plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        plt.yticks(range(len(labels)), labels, fontsize=6)
    else:
        plt.xticks([], [])
        plt.yticks([], [])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_pca_map(df: pd.DataFrame, out_png: str = "pca_map.png", color_col: str = "energy"):
    from sklearn.decomposition import PCA
    feats = df[["tempo","energy","valence","danceability","loudness","acousticness"]].fillna(0.0).to_numpy()
    x = PCA(n_components=2).fit_transform(feats)
    c = df[color_col].to_numpy() if color_col in df.columns else None
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x[:,0], x[:,1], c=c)
    if c is not None:
        plt.colorbar(sc, label=color_col)
    plt.title("2-D PCA map")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_umap_map(df: pd.DataFrame, out_png: str = "umap_map.png", color_col: str = "energy") -> bool:
    try:
        import umap
    except Exception:
        return False
    feats = df[["tempo","energy","valence","danceability","loudness","acousticness"]].fillna(0.0).to_numpy()
    x = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.15).fit_transform(feats)
    c = df[color_col].to_numpy() if color_col in df.columns else None
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x[:,0], x[:,1], c=c)
    if c is not None:
        plt.colorbar(sc, label=color_col)
    plt.title("2-D UMAP map")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True

def write_topk_neighbors(S: np.ndarray, df: pd.DataFrame, k: int = 5, out_csv: str = "topk_neighbors.csv"):
    rows = []
    n = len(df)
    for i in range(n):
        sims = [(j, S[i, j]) for j in range(n) if j != i]
        sims.sort(key=lambda t: t[1], reverse=True)
        for rank, (j, sim) in enumerate(sims[:k], 1):
            rows.append({
                "track": df.loc[i, "name"], "artist": df.loc[i, "artist"],
                "neighbor_rank": rank,
                "neighbor_track": df.loc[j, "name"], "neighbor_artist": df.loc[j, "artist"],
                "similarity": round(float(sim), 4)
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def write_weakest_pairs(S: np.ndarray, df: pd.DataFrame, m: int = 50, out_csv: str = "weakest_pairs.csv"):
    n = len(df)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, S[i, j]))
    pairs.sort(key=lambda x: x[2])  # ascending similarity
    rows = []
    for i, j, sim in pairs[:m]:
        rows.append({
            "a_track": df.loc[i, "name"], "a_artist": df.loc[i, "artist"],
            "b_track": df.loc[j, "name"], "b_artist": df.loc[j, "artist"],
            "similarity": round(float(sim), 4)
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
