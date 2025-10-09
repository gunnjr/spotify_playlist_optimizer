import os
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .demo_dataset import demo_df
from .reporting.visuals import (
    sample_matrix, plot_similarity_heatmap, plot_pca_map, plot_umap_map,
    write_topk_neighbors, write_weakest_pairs
)
from .reporting.html_report import render_report
from .plugins.greedy_two_opt import optimize_cycle, cycle_score

# Optional import of spotipy only when needed (skip in --demo)
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception:
    spotipy = None
    SpotifyOAuth = None

# -----------------------------
# Musical helpers
# -----------------------------
CIRCLE_OF_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
COF_POS = {pc: i for i, pc in enumerate(CIRCLE_OF_FIFTHS_ORDER)}

def key_distance_on_circle(a_key: Optional[int], b_key: Optional[int]) -> int:
    if a_key is None or b_key is None:
        return 6
    a = COF_POS.get(a_key % 12)
    b = COF_POS.get(b_key % 12)
    if a is None or b is None:
        return 6
    d = abs(a - b)
    return min(d, 12 - d)

def mode_distance(a_mode: Optional[int], b_mode: Optional[int]) -> int:
    if a_mode is None or b_mode is None:
        return 1
    return 0 if int(a_mode) == int(b_mode) else 1

def tempo_ratio_delta(a_tempo: Optional[float], b_tempo: Optional[float]) -> float:
    if not a_tempo or not b_tempo:
        return 1.0
    r = min(a_tempo / b_tempo, b_tempo / a_tempo)
    return 1.0 - r

# -----------------------------
# Similarity computation
# -----------------------------
def compute_similarity_matrix(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    n = len(df)
    norm_df = df.copy()
    scalars = ["energy", "valence", "danceability", "loudness", "acousticness"]
    for col in scalars:
        vals = pd.to_numeric(norm_df[col], errors="coerce").to_numpy(dtype=float)
        m = np.nanmedian(vals) if np.isnan(vals).any() else None
        if m is not None:
            vals = np.where(np.isnan(vals), m, vals)
        mn, mx = np.min(vals), np.max(vals)
        norm_df[col] = (vals - mn) / (mx - mn) if mx > mn else 0.0

    tempo = pd.to_numeric(df["tempo"], errors="coerce").to_numpy(dtype=float)
    key = pd.to_numeric(df["key"], errors="coerce").to_numpy(dtype=float)
    mode = pd.to_numeric(df["mode"], errors="coerce").to_numpy(dtype=float)
    energy = norm_df["energy"].to_numpy()
    valence = norm_df["valence"].to_numpy()
    dance = norm_df["danceability"].to_numpy()
    loud = norm_df["loudness"].to_numpy()
    acoustic = norm_df["acousticness"].to_numpy()

    w_tempo = float(weights.get("tempo", 0.20))
    w_keymode = float(weights.get("key_mode", 0.25))
    w_energy = float(weights.get("energy", 0.20))
    w_valence = float(weights.get("valence", 0.15))
    w_dance = float(weights.get("danceability", 0.10))
    w_loud = float(weights.get("loudness", 0.05))
    w_ac = float(weights.get("acousticness", 0.05))

    S = np.zeros((n, n), dtype=float)
    max_dist = math.sqrt(sum([w_tempo, w_keymode, w_energy, w_valence, w_dance, w_loud, w_ac]))
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            dt = tempo_ratio_delta(tempo[i], tempo[j])
            dk = key_distance_on_circle(
                int(key[i]) if not math.isnan(key[i]) else None,
                int(key[j]) if not math.isnan(key[j]) else None
            )
            dk_norm = dk / 6.0
            dm = mode_distance(
                int(mode[i]) if not math.isnan(mode[i]) else None,
                int(mode[j]) if not math.isnan(mode[j]) else None
            )
            de = abs(energy[i] - energy[j])
            dv = abs(valence[i] - valence[j])
            dd = abs(dance[i] - dance[j])
            dl = abs(loud[i] - loud[j])
            da = abs(acoustic[i] - acoustic[j])

            dist = math.sqrt(
                w_tempo * (dt ** 2) +
                w_keymode * (((dk_norm + dm * 0.5) / 1.5) ** 2) +
                w_energy * (de ** 2) +
                w_valence * (dv ** 2) +
                w_dance * (dd ** 2) +
                w_loud * (dl ** 2) +
                w_ac * (da ** 2)
            )
            sim = 1.0 - (dist / max_dist if max_dist > 0 else 0.0)
            S[i, j] = S[j, i] = max(0.0, min(1.0, sim))
    return S

# -----------------------------
# Spotify IO (lazy import)
# -----------------------------
def build_spotify_client(write_back: bool, client_id: Optional[str], redirect_uri: Optional[str]):
    if spotipy is None or SpotifyOAuth is None:
        raise SystemExit("spotipy not installed. Install requirements or use --demo mode.")
    load_dotenv()
    env_client = os.getenv("SPOTIPY_CLIENT_ID")
    env_redirect = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8080/callback")
    cid = client_id or env_client
    ruri = redirect_uri or env_redirect
    if not cid:
        raise SystemExit("Missing SPOTIPY_CLIENT_ID (pass --client-id or set in .env).")

    scopes = ["playlist-read-private", "playlist-read-collaborative"]
    if write_back:
        scopes += ["playlist-modify-private", "playlist-modify-public"]

    auth = SpotifyOAuth(
        client_id=cid,
        redirect_uri=ruri,
        scope=" ".join(scopes),
        cache_path=".spotipyoauthcache",
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth)

def fetch_playlist_tracks(sp, playlist_id: str) -> pd.DataFrame:
    fields = "items(track(id,name,uri,artists(name),album(name,release_date,release_date_precision))),next"
    results = sp.playlist_items(playlist_id, fields=fields, additional_types=["track"], limit=100)
    items = []
    while True:
        for it in results["items"]:
            t = it.get("track")
            if not t or not t.get("id"):
                continue
            items.append({
                "id": t["id"],
                "uri": t.get("uri"),
                "name": t.get("name"),
                "artist": ", ".join(a["name"] for a in t.get("artists", []) if a and a.get("name")),
                "album": (t.get("album") or {}).get("name"),
                "release_date": (t.get("album") or {}).get("release_date"),
            })
        if results.get("next"):
            results = sp.next(results)
        else:
            break
    return pd.DataFrame(items).drop_duplicates(subset=["id"]).reset_index(drop=True)

def fetch_audio_features(sp, ids: List[str]) -> pd.DataFrame:
    feats = []
    for i in range(0, len(ids), 100):
        feats += sp.audio_features(ids[i:i + 100])
    rows = []
    for f in feats:
        if not f or not f.get("id"):
            continue
        rows.append({
            "id": f["id"],
            "tempo": f.get("tempo"),
            "key": f.get("key"),
            "mode": f.get("mode"),
            "energy": f.get("energy"),
            "valence": f.get("valence"),
            "danceability": f.get("danceability"),
            "loudness": f.get("loudness"),
            "acousticness": f.get("acousticness"),
        })
    return pd.DataFrame(rows)

def write_back_playlist(sp, user_id: str, name: str, uris_ordered: List[str]) -> str:
    pl = sp.user_playlist_create(user=user_id, name=name, public=False,
                                 description="Optimized loop order")
    pid = pl["id"]
    for i in range(0, len(uris_ordered), 100):
        sp.playlist_add_items(pid, uris_ordered[i:i+100])
    return pid

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze and optimize Spotify playlist order with visual diagnostics.")
    # Core
    ap.add_argument("--playlist-id", help="Spotify playlist ID or URI (omit with --demo)")
    ap.add_argument("--out", default="ordered_playlist.csv", help="Output CSV path (order + edge sims)")
    ap.add_argument("--dry-run", action="store_true", help="Analyze/visualize only; skip reordering")
    ap.add_argument("--write-back", action="store_true", help="Create a new Spotify playlist with optimized order")
    ap.add_argument("--new-playlist-name", default="Optimized Loop", help="Name for the new playlist if --write-back")
    ap.add_argument("--starts", type=int, default=10, help="Random starts for multi-start greedy")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    # Weights
    ap.add_argument("--w-tempo", type=float, default=0.20)
    ap.add_argument("--w-keymode", type=float, default=0.25)
    ap.add_argument("--w-energy", type=float, default=0.20)
    ap.add_argument("--w-valence", type=float, default=0.15)
    ap.add_argument("--w-dance", type=float, default=0.10)
    ap.add_argument("--w-loud", type=float, default=0.05)
    ap.add_argument("--w-acoustic", type=float, default=0.05)
    # Visualization & reports
    ap.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    ap.add_argument("--viz-sample", type=int, default=80, help="Max songs to include in heatmap")
    ap.add_argument("--viz-out", default="similarity_heatmap.png", help="Heatmap output PNG")
    ap.add_argument("--pca", action="store_true", help="Generate PCA map")
    ap.add_argument("--umap", action="store_true", help="Generate UMAP map (requires umap-learn)")
    ap.add_argument("--pca-out", default="pca_map.png", help="PCA output PNG")
    ap.add_argument("--umap-out", default="umap_map.png", help="UMAP output PNG")
    ap.add_argument("--topk", type=int, default=5, help="Neighbors per song (0=skip)")
    ap.add_argument("--topk-out", default="topk_neighbors.csv", help="Top-k neighbors CSV")
    ap.add_argument("--weakest", type=int, default=50, help="How many weakest pairs to save (0=skip)")
    ap.add_argument("--weakest-out", default="weakest_pairs.csv", help="Weakest pairs CSV")
    ap.add_argument("--html-report", default=None, help="Output HTML report path (e.g., report.html)")
    # Credentials
    ap.add_argument("--client-id", default=None, help="Spotify client ID (overrides env/.env)")
    ap.add_argument("--redirect-uri", default=None, help="Spotify redirect URI (overrides env/.env)")
    # Demo
    ap.add_argument("--demo", action="store_true", help="Use a synthetic dataset (no Spotify calls)")

    args = ap.parse_args()

    # Prepare data frame
    if args.demo:
        df = demo_df(seed=args.seed, n=10)
        playlist_name = "Demo Dataset"
        sp = None
    else:
        if not args.playlist_id:
            raise SystemExit("Provide --playlist-id or use --demo.")
        sp = build_spotify_client(write_back=args.write_back, client_id=args.client_id, redirect_uri=args.redirect_uri)
        print("Fetching playlist tracks...")
        df_tracks = fetch_playlist_tracks(sp, args.playlist_id)
        if df_tracks.empty:
            raise SystemExit("No retrievable tracks.")
        print(f"Found {len(df_tracks)} tracks. Fetching audio features...")
        feats = fetch_audio_features(sp, df_tracks["id"].tolist())
        df = df_tracks.merge(feats, on="id", how="left")
        df = df.dropna(subset=["tempo","key","mode","energy","valence","danceability","loudness","acousticness"]).reset_index(drop=True)
        playlist_name = args.playlist_id

    # Compute similarity
    weights = {
        "tempo": args.w_tempo,
        "key_mode": args.w_keymode,
        "energy": args.w_energy,
        "valence": args.w_valence,
        "danceability": args.w_dance,
        "loudness": args.w_loud,
        "acousticness": args.w_acoustic,
    }
    print("Computing similarity matrix...")
    S = compute_similarity_matrix(df, weights=weights)

    # Visualizations / diagnostics
    heatmap_path = pca_path = umap_path = None
    if not args.no-viz if False else True:
        pass  # placeholder to avoid syntax highlight issue

    if not args.no_viz:
        S_sub, df_sub, _ = sample_matrix(S, df, max_n=args.viz_sample, seed=args.seed)
        labels = (df_sub["artist"] + " — " + df_sub["name"]).tolist()
        plot_similarity_heatmap(S_sub, labels, out_png=args.viz_out)
        heatmap_path = args.viz_out
        if args.pca:
            plot_pca_map(df, out_png=args.pca_out, color_col="energy")
            pca_path = args.pca_out
        if args.umap:
            if plot_umap_map(df, out_png=args.umap_out, color_col="energy"):
                umap_path = args.umap_out
            else:
                print("UMAP not available. Install `umap-learn` to enable.")

        if args.topk > 0:
            write_topk_neighbors(S, df, k=args.topk, out_csv=args.topk_out)
        if args.weakest > 0:
            write_weakest_pairs(S, df, m=args.weakest, out_csv=args.weakest_out)

    avg_sim = None
    tour = None

    if not args.dry_run:
        print("Optimizing loop order (greedy + 2-opt plugin)...")
        tour = optimize_cycle(S, starts=args.starts, seed=args.seed)
        avg_sim = cycle_score(S, tour)
        trans_sims = [S[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))]
        ordered = df.iloc[tour].copy().reset_index(drop=True)
        ordered.insert(0, "order_index", range(len(ordered)))
        ordered["transition_similarity_to_next"] = trans_sims
        ordered.to_csv(args.out, index=False)
        print(f"Saved ordered CSV → {args.out}")
        print(f"Loop smoothness → avg: {avg_sim:.4f} | min: {min(trans_sims):.4f} | max: {max(trans_sims):.4f}")
        if args.write_back and sp is not None:
            print("Writing back to new Spotify playlist...")
            user_id = sp.current_user()["id"]
            pid = write_back_playlist(sp, user_id, args.new_playlist_name, ordered["uri"].tolist())
            print(f"Created playlist: https://open.spotify.com/playlist/{pid}")
    else:
        print("Dry-run: analysis complete; skipped reordering.")

    # HTML summary report
    if args.html_report:
        render_report(
            out_html=args.html_report,
            playlist_name=playlist_name,
            n_tracks=len(df),
            avg_sim=avg_sim,
            heatmap_path=heatmap_path,
            pca_path=pca_path,
            umap_path=umap_path,
            topk_csv=args.topk_out if args.topk > 0 else None,
            weakest_csv=args.weakest_out if args.weakest > 0 else None,
        )
        print(f"Wrote HTML report → {args.html_report}")

if __name__ == "__main__":
    main()
