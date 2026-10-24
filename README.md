# Hit Factors: what makes a song a hit?

> **Version 1, working draft.** This README summarizes how we currently see the project. Everything is open to discussion within the group; open questions are listed at the end.

## 1. Goal

Statistically and textually analyze thousands of songs to identify the factors that make a song a hit: its musical characteristics, its lyrics and its release context.

The project answers two questions:

1. **Hit or miss?** What distinguishes a song that enters the Billboard Hot 100 from a song that never does? And did these factors change with the rise of streaming?
2. **Among the songs that charted, what makes a bigger hit?** What explains why a song climbs higher or stays longer on the chart?

## 2. Definition of success

**A song is a "hit" if it entered the Billboard Hot 100 at least once.**

- This is a binary definition (yes / no), easy to model and to explain.
- It requires songs that **did not** chart as well, released over the same period. Studying only hits would not allow any conclusion (survivorship bias).
- Billboard only measures the **US** market: a limitation to acknowledge in the presentation.
- We **do not** use Spotify popularity as a measure of success: it is a snapshot taken at collection time, which favors recent songs.

## 3. Scope

- **Period: 2000 to 2021.** The available Spotify datasets were collected before Spotify closed access to audio features (late 2024). 2022 is excluded, because songs released just before collection had not yet had time to chart.
- **Two eras, one model per era**, then a comparison of which variables matter in each:
  - **2000-2011**: download / radio era;
  - **2012-2021**: streaming era (Billboard has included streaming in its chart formula since 2012).
- To keep the comparison fair, we keep a similar proportion of "yes" and "no" from one year to the next.

## 4. Building the dataset

We start from **a single base dataset** (Kaggle Spotify) and add a `hit` column. Each row is a song.

1. Take all Spotify songs released between 2000 and 2021.
2. Match them with Billboard on **title + main artist**: `hit = 1` if the song charted on the Hot 100, `0` otherwise.
3. Enrich each song with its lyrics (Genius) and its release information (MusicBrainz).

Why not two separate datasets, one for hits and one for the rest? Billboard contains no duration, genre or audio features. And if the two groups came from two different sources, the model would learn to recognize the source rather than success.

**Matching.** There is no common identifier across sources. We normalize title and artist (lowercase, no accents, no "feat.", no "- Remastered"…), then use fuzzy matching (for example with `rapidfuzz`). ISRC codes can help when available.

**Expected imbalance.** There will be many more "no" than "yes". This is normal; we handle it during modeling and evaluate with F1-score or AUC, not accuracy.

## 5. Output and inputs

### Output (what we want to predict)

| Question | Output | Type | Songs included |
|---|---|---|---|
| 1. Hit or miss | `hit`: 1 if the song entered the Hot 100, 0 otherwise | Binary classification | All songs, one model per era |
| 2. Hit intensity | `peak_pos`: best chart position reached | Regression | Charted songs only |
| 2. Hit intensity | `wks_on_chart`: number of weeks on the chart | Regression | Charted songs only |

### Inputs (song characteristics)

**Song DNA** (Kaggle Spotify, MusicBrainz)
- duration;
- genre (to be grouped into 8 to 10 broad genres);
- explicit content;
- audio features: tempo, energy, danceability, valence (musical positivity), loudness, acousticness, speechiness, instrumentalness, major / minor mode;
- independent or signed artist (derived from the label).

**Lyrics** (Kaggle Genius, variables computed with NLP)
- positivity / melancholy score;
- main theme;
- vocabulary richness;
- word count;
- presence of profanity.

**Release context** (MusicBrainz, Kaggle)
- release month or season;
- day of the week (note: since July 2015, global releases happen on Fridays);
- presence of a featured artist and number of artists.

**Example row:** 3 min 20 s, pop, 120 BPM, signed label, positive lyrics, released on a Friday in June, 1 featured artist → `hit = 1`.

### Never use as an input

Any information that results from success itself: Spotify popularity, Genius views, stream counts, and the Billboard columns (`peak_pos`, `wks_on_chart`, `chart_week`). Otherwise the model "cheats" by indirectly seeing the answer.

## 6. Datasets and extracted information

> The column names below are indicative. They vary depending on the dataset version downloaded: **check them with `df.columns`** when opening each file.

### Kaggle Spotify (base dataset)

For example "Spotify Tracks Dataset" (maharshipandya, 114k tracks, 125 genres) or "Spotify 1.2M+ Songs" (rodolfofigueroa). CSV download with a free Kaggle account.

| Role | Columns |
|---|---|
| Matching | `track_name`, `artists` |
| Inputs | `duration_ms`, `track_genre`, `explicit`, `danceability`, `energy`, `valence`, `tempo`, `loudness`, `acousticness`, `speechiness`, `instrumentalness`, `mode` |
| Exclude | `popularity` (depends on success) |

### Billboard Hot 100 (`utdata/rwd-billboard-data` dataset)

Every weekly chart since 1958, updated weekly. Direct CSV: `https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/data-out/hot-100-current.csv`

| Role | Columns |
|---|---|
| Matching | `title`, `performer` |
| Output, question 1 | `hit` = 1 if the song appears in the file |
| Output, question 2 | `peak_pos`, `wks_on_chart` |
| Descriptive analysis | `chart_week` (chart entry date, which can also approximate the release date when MusicBrainz does not provide it) |

### Kaggle "Genius Song Lyrics" (carlosgdcj)

About 5 million songs with lyrics (~9 GB).

| Role | Columns |
|---|---|
| Matching | `title`, `artist` |
| NLP variables | `lyrics` |
| Filter | `language` (keep English: sentiment tools such as VADER are English-only) |
| Direct inputs | `features` (featured artists), `year` |
| Exclude | `views` (depends on success) |

Lyrics are copyrighted: we analyze them, we do not republish them. Remember to strip tags such as `[Chorus]`, `[Verse 1]`.

### MusicBrainz (open API, CC0)

REST API `musicbrainz.org/ws/2/` (1 request per second, User-Agent required), or full database dump.

| Role | Fields |
|---|---|
| Inputs | `date` (exact release date, missing from most Spotify datasets), `label`, `artist-credit` (to confirm featured artists) |
| To compute | independent or signed, from the label |

The one-request-per-second limit makes this step slow: start it early.

### Sources ruled out

- **Spotify API**: since 2024-2026, no more audio features, popularity or label data, and a Premium account is required.
- **Scraping Genius**: the official API does not return lyrics, and scraping violates the terms of service.

Full source details are in `sources/sources_donnees.md`.

## 7. Planned analyses

1. **Exploratory analysis**: variable distributions, differences between charted and non-charted songs, trends over time (are hits getting shorter? is rap taking over?).
2. **Question 1, per era**: one classification model for 2000-2011, one for 2012-2021, then a comparison of variable importance before and after streaming.
3. **Question 2**: among charted songs, regression on `peak_pos` and `wks_on_chart` with the same inputs.
4. **Lyrics**: comparison between lyrics sentiment and musical valence.

## 8. Pitfalls to watch

- **Data leakage**: no success-dependent variable as an input (see section 5).
- **Matching**: manually check a sample of matches to measure fuzzy-matching errors.
- **Dates**: Spotify often gives the album or reissue date; prefer MusicBrainz. Exclude year-only dates from the day-of-week analysis.
- **Genre**: often assigned to the artist rather than the song.
- **Independent / signed**: the hardest variable in the project (many "indies" are distributed by a major). Do it on a sample and document uncertain cases.

## 9. Open questions

- How the work is split among group members.
- The exact Kaggle Spotify dataset to use (size, coverage of 2000-2021).
- The exact method to classify a label as independent or signed.
- Which models to use (logistic regression, random forest, gradient boosting…).
