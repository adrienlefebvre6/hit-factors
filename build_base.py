# Builds the project dataset (one row = one song, hit column = 0/1) for 2000-2020.
#
# Expected files in CSV/:
#   hot-100-2000.csv    weekly Billboard Hot 100 since 2000
#   audio_features.csv  TidyTuesday 2021-09-14 (Sean Miller): audio features + Spotify id + genres of Hot 100 songs
#                       https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-09-14/audio_features.csv
#   tracks.csv          Kaggle "Spotify Dataset 1921-2020, 600k+ Tracks" (yamaerenay)
#   artists.csv         same Kaggle dataset (genres per artist)
#
# Usage: python build_base.py
# Outputs: CSV/base_full.csv (all non-hits) and CSV/base_sample.csv (N_NEG non-hits per hit, same year)
import re
import unicodedata
import pandas as pd

FOLDER = "CSV/"
YEARS = (2000, 2020)
N_NEG = 5

AUDIO = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "time_signature", "duration_ms", "explicit"]


def norm(s):
    """Normalized title or artist: lowercase, no accents, no feat./parentheses/' - Remastered'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.split(r"\s+(feat\.?|featuring|ft\.?)\s+", s)[0]
    s = re.sub(r"\(.*?\)|\[.*?\]| - .*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def main_artist_bb(a):
    """Billboard: 'Beyonce Featuring The Weeknd' -> 'beyonce'."""
    return norm(re.split(r"\s+(?:featuring|feat\.?|ft\.?|&|x|with|and)\s+|,", str(a), flags=re.I)[0])


def first_in_list(a):
    """Spotify: "['Beyoncé', 'The Weeknd']" -> 'Beyoncé'."""
    m = re.findall(r"['\"](.+?)['\"](?:,|\])", str(a))
    return m[0] if m else str(a)


# 1. Billboard: one row per song, with its chart information
bb = pd.read_csv(FOLDER + "hot-100-2000.csv")
last_week = bb["date"].max()
bb = bb.sort_values("date").groupby(["song", "artists"]).agg(
    first_chart_week=("date", "first"),
    last_chart_week=("date", "last"),
    peak_rank=("rank", "min"),
    weeks_on_chart=("rank", "size"),
    weeks_before=("weeks-on-board", "first"),
).reset_index()
bb["year"] = bb["first_chart_week"].str[:4].astype(int)
bb = bb[bb["year"].between(*YEARS)]
# edge cases, flagged rather than removed
bb["charted_before_2000"] = (bb["first_chart_week"] == "2000-01-01") & (bb["weeks_before"] > 1)
bb["still_charting"] = bb["last_chart_week"] == last_week
bb["key_full"] = bb["song"].map(norm) + "|" + bb["artists"].map(norm)
bb["key_main"] = bb["song"].map(norm) + "|" + bb["artists"].map(main_artist_bb)

# 2. Hits: Billboard + audio_features (same Billboard source, joined on title + full performer)
af = pd.read_csv(FOLDER + "audio_features.csv")
af = af[af["danceability"].notna()].drop_duplicates("song_id")
af["key_full"] = af["song"].map(norm) + "|" + af["performer"].map(norm)
af = af.drop_duplicates("key_full")
af = af.rename(columns={"spotify_track_id": "id", "spotify_track_duration_ms": "duration_ms",
                        "spotify_track_explicit": "explicit", "spotify_genre": "genres"})
hits = bb.merge(af[["key_full", "id", "genres"] + AUDIO], on="key_full", how="inner")
hits = hits.sort_values("peak_rank").drop_duplicates("id")   # same track under two credits (e.g. Levitating)
hits = hits.rename(columns={"artists": "artist"})
hits["hit"] = 1
print(f"Hits {YEARS[0]}-{YEARS[1]}: {len(hits)} / {len(bb)} Billboard songs ({len(hits) / len(bb):.0%})")

# 3. Non-hits: 600k tracks released in the period that never charted
tr = pd.read_csv(FOLDER + "tracks.csv")
tr["year"] = pd.to_numeric(tr["release_date"].astype(str).str[:4], errors="coerce")
tr = tr[tr["year"].between(*YEARS)].copy()
tr["artist"] = tr["artists"].map(first_in_list)
tr["artist_id"] = tr["id_artists"].map(first_in_list)
tr["key_main"] = tr["name"].map(norm) + "|" + tr["artist"].map(norm)
# remove anything that looks like a hit (same id, or same title + main artist),
# including hits without audio features: otherwise they would wrongly count as non-hits
charted = tr["id"].isin(hits["id"]) | tr["key_main"].isin(bb["key_main"])
print(f"600k tracks {YEARS[0]}-{YEARS[1]}: {len(tr)} | removed as charted: {int(charted.sum())}")
neg = tr[~charted].drop_duplicates("key_main").rename(columns={"name": "song"})

# non-hit genres: Spotify genres of the main artist (same kind of data as spotify_genre for hits)
art = pd.read_csv(FOLDER + "artists.csv", usecols=["id", "genres"]).rename(columns={"id": "artist_id"})
neg = neg.merge(art, on="artist_id", how="left")
neg["hit"] = 0

# 4. Final table (no popularity, followers or views: leakage variables)
COLS = ["id", "song", "artist", "year", "hit", "genres"] + AUDIO
HIT_COLS = ["first_chart_week", "peak_rank", "weeks_on_chart", "charted_before_2000", "still_charting"]
base = pd.concat([hits[COLS + HIT_COLS], neg[COLS + ["release_date"]]], ignore_index=True)
base["year"] = base["year"].astype(int)
base["explicit"] = base["explicit"].astype(str).str.lower().isin(["true", "1"]).astype(int)

# 5. Final cleaning
# outlier audio values (empty tracks, interludes, tracks longer than 10 min)
outlier = (base["tempo"] == 0) | (base["time_signature"] == 0) | ~base["duration_ms"].between(60_000, 600_000)
print(f"Rows removed (audio outliers): {int(outlier.sum())} incl. {int(base.loc[outlier, 'hit'].sum())} hits")
base = base[~outlier]

# genre families (the first family with a keyword found in the artist's genres wins)
FAMILIES = [
    ("latin", ["latin", "reggaeton", "salsa", "bachata", "cumbia", "banda", "mexican"]),
    ("rap/hip-hop", ["hip hop", "rap", "trap", "drill", "grime"]),
    ("country", ["country", "bluegrass"]),
    ("r&b/soul", ["r&b", "soul", "funk", "motown"]),
    ("rock", ["rock", "metal", "punk", "grunge", "emo", "hardcore"]),
    ("pop", ["pop", "boy band", "girl group"]),
    ("electro/dance", ["edm", "house", "techno", "electro", "dance", "trance", "dubstep", "drum and bass"]),
    ("jazz/blues", ["jazz", "blues", "swing"]),
    ("folk/acoustic", ["folk", "singer-songwriter", "acoustic"]),
    ("classical/soundtrack", ["classical", "orchestra", "soundtrack", "score", "opera"]),
]


def family(genres):
    g = str(genres).lower()
    if g in ("", "nan", "[]"):
        return "unknown"
    for name, words in FAMILIES:
        if any(w in g for w in words):
            return name
    return "other"


base["genre_family"] = base["genres"].map(family)
base.to_csv(FOLDER + "base_full.csv", index=False)

sample = pd.concat([
    base[base["hit"] == 1],
    base[base["hit"] == 0].groupby("year", group_keys=False).apply(
        lambda g: g.sample(min(len(g), N_NEG * int((base["hit"] == 1)[base["year"] == g.name].sum())), random_state=0)),
], ignore_index=True)
sample.to_csv(FOLDER + "base_sample.csv", index=False)

print(f"base_full.csv: {len(base)} rows incl. {int(base['hit'].sum())} hits")
print(f"base_sample.csv: {len(sample)} rows incl. {int(sample['hit'].sum())} hits")
print("Genre available: hits %.0f %%, non-hits %.0f %%" % (
    100 * (base.loc[base.hit == 1, "genres"].fillna("[]") != "[]").mean(),
    100 * (base.loc[base.hit == 0, "genres"].fillna("[]") != "[]").mean()))
print(base.groupby("year")["hit"].agg(rows="size", hits="sum").to_string())
