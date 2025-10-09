#!/usr/bin/env bash
set -euo pipefail

# Dry-run analysis on a real Spotify playlist (no reordering).
# Usage: ./scripts/playlist_dry_run.sh <PLAYLIST_ID_OR_URI>

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <PLAYLIST_ID_OR_URI>"
  exit 1
fi

PLAYLIST_ID="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$REPO_ROOT/output"
mkdir -p "$OUT_DIR"

# If your venv isn't already active, uncomment:
# source .venv/bin/activate

python -m src.spotify_playlist_optimizer \
  --playlist-id "$PLAYLIST_ID" \
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

echo "✅ Dry run complete for playlist: $PLAYLIST_ID. See $OUT_DIR"

