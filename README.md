# spotify_playlist_optimizer

Privacy-first, local Python toolkit to **analyze and optimize Spotify playlists** for continuous flow.
Fetches audio features (tempo, energy, valence, etc.), computes song-to-song similarity, and can:
- Visualize how songs relate to each other
- Report cohesion and outliers
- Optimize play order for seamless transitions (loop-friendly)

## 🧭 Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a Spotify app (no secret needed, PKCE flow):
- https://developer.spotify.com/dashboard → create app
- Add redirect URI: `http://localhost:8080/callback`
- Copy **Client ID**

Put credentials in `.env` (or pass via CLI in the next step):
```bash
SPOTIPY_CLIENT_ID="your_client_id"
SPOTIPY_REDIRECT_URI="http://localhost:8080/callback"
```

## 🧪 Core commands (preview of v0.1.0 features)

- `--dry-run` – analyze & visualize only (no reordering)
- `--out ordered.csv` – export optimized loop order
- `--write-back` – create a new Spotify playlist in that order
- `--viz-*` – control visualizations (heatmap, neighbors, weakest pairs)
- `--pca` / `--umap` – generate 2‑D maps (playlist structure)
- `--html-report` – one-page HTML summary combining plots + stats

## ⚙️ Credentials & config

Credential loading order (highest precedence first):
1. CLI flags: `--client-id`, `--redirect-uri`
2. `.env` file
3. Environment variables
4. (Planned) `~/.spotify_optimizer.toml`

## 🧩 Extensible cycle algorithms

Playlist ordering uses pluggable “cycle constructors”:
- Default: Greedy + 2‑opt (symmetric)
- Future: annealing / GA / cluster-first variants

## 📊 Offline demo

Run `--demo` to test with a synthetic 10‑track dataset (no Spotify auth).

## 🧾 Cite

> Gunn, J. (2025). *spotify_playlist_optimizer*: Analyze and optimize looped playlist order.  
> GitHub. https://github.com/yourusername/spotify_playlist_optimizer

## 🛠️ License

MIT © 2025 John Gunn
