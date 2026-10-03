# Données

Les données ne sont **pas** versionnées dans Git (trop lourdes, et les paroles sont protégées par le droit d'auteur). Chacun les télécharge dans `data/raw/` avec les noms ci-dessous, pour que les notebooks marchent chez tout le monde.

| Dossier | Contenu |
|---|---|
| `raw/` | Fichiers tels que téléchargés. On ne les modifie jamais. |
| `interim/` | Une table nettoyée par source (Parquet). |
| `processed/` | `morceaux.parquet` : table fusionnée produite par `notebooks/00_cible_et_jointure.ipynb`, utilisée par tous les axes. |

## Sources et noms de fichiers attendus

| Fichier dans `raw/` | Source | Comment l'obtenir |
|---|---|---|
| `billboard_hot100.csv` | [utdata/rwd-billboard-data](https://github.com/utdata/rwd-billboard-data), Hot 100 hebdomadaire depuis 1958 | Automatique : `hit_factors.data.load_billboard()` le télécharge au premier appel. |
| `spotify_tracks.csv` | Kaggle [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (maharshipandya) | `kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/raw --unzip`, puis renommer `dataset.csv`. |
| `genius_song_lyrics.csv` | Kaggle [Genius Song Lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) (carlosgdcj), ~9 Go | `kaggle datasets download -d carlosgdcj/genius-song-lyrics-with-language-information -p data/raw --unzip`, puis renommer `song_lyrics.csv`. |
| `musicbrainz/` | API [MusicBrainz](https://musicbrainz.org/doc/MusicBrainz_API) (CC0) : labels, dates de sortie, crédits | Par script, avec `musicbrainzngs`. **1 requête/s** et User-Agent obligatoire (voir `.env.example`). À lancer tôt, c'est long. |

Le détail des sources, de leurs limites et de ce qui a été écarté est dans le fichier partagé du projet `sources/sources_donnees.md`.

**Ce qu'on n'utilise pas :** l'API Spotify (audio features retirées en 2024, popularité et label retirés en 2026) et le scraping de Genius (interdit par ses CGU).

## Clé Kaggle

Créer un token sur kaggle.com > Settings > API, puis placer `kaggle.json` dans `~/.kaggle/` (jamais dans le dépôt).
