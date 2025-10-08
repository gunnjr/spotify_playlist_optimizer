# `src/` overview

This package analyzes Spotify playlists, visualizes similarity, and (optionally) computes an optimized **loop** order so the wrap from last → first feels smooth.

## How it fits together

```
Spotify API / Demo → dataframe (tracks + features)
                 ↓
      similarity matrix S (0..1)
                 ↓
     visualizations + reports (optional)
                 ↓
    cycle constructor plugin (optimize order)
                 ↓
  ordered CSV (+ optional write-back to Spotify)
```

## Modules and roles

| File | Purpose |
|------|----------|
| `spotify_playlist_optimizer.py` | **Main CLI module.** Loads credentials, fetches tracks/features (or demo data), computes similarity, runs visualizations, calls a cycle constructor plugin, writes CSV and optional HTML report, and can write a new Spotify playlist. |
| `demo_dataset.py` | Generates a small synthetic dataset (10 tracks) so you can run analysis and visualizations without Spotify auth (`--demo`). |
| `plugins/greedy_two_opt.py` | Default **cycle constructor**: nearest-neighbor greedy tour + **2-opt** improvement. Provides `optimize_cycle()` and `cycle_score()`. Serves as the reference plugin. |
| `reporting/visuals.py` | Visualization utilities: similarity **heatmap**, **PCA** and (optional) **UMAP** 2-D maps; CSV writers for top-k neighbors and weakest pairs; matrix sampling for readable plots. |
| `reporting/html_report.py` | Simple HTML report generator that stitches together plots + CSV references into one page. |

## Key concepts

- **Similarity matrix `S`**: pairwise track similarity in [0..1], higher = smoother transition. Built from tempo ratio, key/mode distance (circle of fifths), energy, valence, danceability, loudness, acousticness.
- **Cycle constructor (plugin)**: given `S`, returns a loop order (permutation) that maximizes average adjacent similarity (including wrap).

## Common CLI entry points (from repo root)

- **Dry-run (no reordering):**

  ```bash
  python -m src.spotify_playlist_optimizer --playlist-id YOUR_PLAYLIST --dry-run --pca --html-report report.html
  ```

- **Demo, no Spotify auth:**

  ```bash
  python -m src.spotify_playlist_optimizer --demo --dry-run --pca --umap --html-report report.html
  ```

- **Optimize & export order:**

  ```bash
  python -m src.spotify_playlist_optimizer --playlist-id YOUR_PLAYLIST --out ordered.csv
  ```

## Adding a new cycle plugin

1. Create a new file in `src/plugins/`, e.g. `annealing.py`.
2. Implement a function with this signature:

   ```python
   def optimize_cycle(S: np.ndarray, starts: int = 10, seed: int = 42) -> list[int]:
       ...
   ```

   (Optional) export a `cycle_score(S, tour)` if your scoring differs.
3. Import and wire it in `spotify_playlist_optimizer.py` behind a flag like `--algo annealing`.

## Extensibility notes

- **Features & weights**: tweak in CLI flags or extend the feature set (e.g., lyrical sentiment) before computing `S`.
- **Visuals**: add more plots to `reporting/visuals.py` and link them in the HTML report.
- **Credentials**: provided via CLI → `.env` → environment (in that order).
