#!/usr/bin/env bash
set -euo pipefail

# Run demo dry-run with visualizations, writing everything to ./output

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$REPO_ROOT/output"
mkdir -p "$OUT_DIR"

# If your venv isn't already active, uncomment and point to it:
# source .venv/bin/activate

python -m src.spotify_playlist_optimizer \
  --demo \
  --dry-run \
  --pca \
  --umap \
  --viz-out "$OUT_DIR/similarity_heatmap.png" \
  --pca-out "$OUT_DIR/pca_map.png" \
  --umap-out "$OUT_DIR/umap_map.png" \
  --topk 5 \
  --topk-out "$OUT_DIR/topk_neighbors.csv" \
  --weakest 50 \
  --weakest-out "$OUT_DIR/weakest_pairs.csv" \
  --html-report "$OUT_DIR/report.html"

echo "✅ Demo dry-run complete. See $OUT_DIR"

