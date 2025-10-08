import numpy as np
import pandas as pd

def demo_df(seed: int = 42, n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"Demo Track {i+1}" for i in range(n)]
    artists = [f"Demo Artist {i%3+1}" for i in range(n)]
    # Generate plausible feature ranges
    tempo = rng.uniform(80, 140, n)
    key = rng.integers(0, 12, n)
    mode = rng.integers(0, 2, n)  # 0 minor, 1 major
    energy = rng.uniform(0.2, 0.9, n)
    valence = rng.uniform(0.2, 0.9, n)
    danceability = rng.uniform(0.3, 0.9, n)
    loudness = rng.uniform(-12, -3, n)  # dB
    acousticness = rng.uniform(0.0, 0.8, n)

    df = pd.DataFrame({
        "id": [f"demo{i}" for i in range(n)],
        "uri": [f"demo:track:{i}" for i in range(n)],
        "name": names,
        "artist": artists,
        "album": ["Demo Album"]*n,
        "release_date": ["2020-01-01"]*n,
        "tempo": tempo,
        "key": key,
        "mode": mode,
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "loudness": loudness,
        "acousticness": acousticness,
    })
    return df
