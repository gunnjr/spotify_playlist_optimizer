#!/usr/bin/env bash
set -euo pipefail

# Runs the optimizer on the built-in demo dataset (no Spotify auth).
# Writes all artifacts to ./output

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$REPO_ROOT/output"
mkdir -p "$OUT_DIR"

# If your venv isn't already active, uncomment:
# source .venv/bin/activate

python -m src.spotify_playlist_optimizer \
  --demo \
  --pca \
  --umap \
  --viz-out "$OUT_DIR/similarity_heatmap.png" \
  --pca-out "$OUT_DIR/pca_map.png" \
  --umap-out "$OUT_DIR/umap_map.png" \
  --topk 5 \
  --topk-out "$OUT_DIR/topk_neighbors.csv" \
  --weakest 50 \
  --weakest-out "$OUT_DIR/weakest_pairs.csv" \
  --html-report "$OUT_DIR/report.html" \
  --out "$OUT_DIR/ordered.csv"

echo "✅ Demo optimize complete. See $OUT_DIR"

