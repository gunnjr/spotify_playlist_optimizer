#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python - <<'PY'
import os, sys, pathlib
print("CWD:", os.getcwd())
env_path = pathlib.Path(".env")
print(".env present:", env_path.is_file(), f"({env_path.resolve()})")

try:
    from dotenv import load_dotenv
except Exception:
    print("❌ python-dotenv not installed. Run: pip install python-dotenv", file=sys.stderr)
    raise

load_dotenv(dotenv_path=env_path, override=True)
print("SPOTIPY_CLIENT_ID loaded:", bool(os.getenv("SPOTIPY_CLIENT_ID")))
print("SPOTIPY_REDIRECT_URI:", os.getenv("SPOTIPY_REDIRECT_URI"))

cid = os.getenv("SPOTIPY_CLIENT_ID")
ruri = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8080/callback")
if not cid:
    print("❌ Missing SPOTIPY_CLIENT_ID (set in .env or environment).", file=sys.stderr)
    sys.exit(1)

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception:
    print("❌ spotipy not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    raise

scopes = "playlist-read-private playlist-read-collaborative"
auth = SpotifyOAuth(
    client_id=cid,
    redirect_uri=ruri,
    scope=scopes,
    cache_path=".spotipyoauthcache",
    open_browser=True,
)
sp = spotipy.Spotify(auth_manager=auth)
me = sp.current_user()
print(f"✅ Auth OK. User: {me.get('display_name') or me['id']}")
PY
