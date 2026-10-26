"""
normalize1.py - alternative normalization + match count between base_full.csv and lyr, in DuckDB.

Normalization (from build_base.py): a song/artist is reduced to lowercase [a-z0-9] only,
with accents removed and "feat./featuring/ft." parts, (...), [...] and " - Remastered"
dropped. Each row gets up to two keys:
    norm(title) | norm(artist)            artist without its feat. part
    norm(title) | main_artist(artist)     artist cut at the first  &  x  with  and  ,  feat
A base row matches a lyr row when they share a key (keys with an empty part are ignored).

Nothing is merged here. It builds the key tables, a `matches1` table
(base_id, lyr_id, year_diff) and prints the match counts. Reuses songs.duckdb:
the existing `lyr` table (no lyrics text, just id/src_id/song/artist/year/features) is reused;
the `base` table is created from base_full.csv.

Usage (notebook):  import normalize1 as n1;  con = n1.run()
Later:  n1.unmatched(con)  to look at misses,  n1.report(con, ...) to re-count.
"""
import re
import unicodedata

import normalize as n     

DB_PATH = "songs.duckdb"
MEMORY_LIMIT = "4GB"
BASE_CSV = "./data/base_full.csv"
BASE_COLS = {"id": "src_id", "song": "song", "artist": "artist", "year": "year", "hit": "hit"}

LYR_CSV = "./data/song_lyrics.csv"
LYR_COLS = {"id": "src_id", "title": "song", "artist": "artist", "year": "year", "features": "features"}

def norm(s):
    """Normalized title or artist: lowercase, no accents, no feat./parentheses/' - Remastered'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.split(r"\s+(feat\.?|featuring|ft\.?)\s+", s)[0]
    s = re.sub(r"\(.*?\)|\[.*?\]| - .*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def main_artist_bb(a):
    """Billboard: 'Beyonce Featuring The Weeknd' -> 'beyonce'."""
    return norm(re.split(r"\s+(?:featuring|feat\.?|ft\.?|&|x|with|and)\s+|,", str(a), flags=re.I)[0])


def register(con):
    """Make the two functions callable from SQL (needed once per connection)."""
    for name, fn in (("nrm_key", norm), ("nrm_main", main_artist_bb)):
        try:
            con.remove_function(name)
        except Exception:
            pass
        con.create_function(name, fn, ["VARCHAR"], "VARCHAR")


def step_load(con):
    print("== load")
    n.load_csv(con, "base", BASE_CSV, BASE_COLS)
    d = con.execute("SELECT count(*) - count(DISTINCT src_id) FROM base").fetchone()[0]
    if d:
        print(f"  warning: {d:,} duplicated ids in base_full (ids kept, matching is unaffected)")
    if n._table_exists(con, "lyr"):
        cols = n._columns(con, "lyr")
        need = {"id", "song", "artist", "year"}
        if not need <= set(cols):
            raise ValueError(f"existing lyr table lacks {need - set(cols)}; reload it")
        c = con.execute("SELECT count(*) FROM lyr").fetchone()[0]
        print(f"  lyr: reusing existing table ({c:,} rows)")
    else:
        n.load_csv(con, "lyr", LYR_CSV, LYR_COLS)


def build_keys(con, table):
    """{table}_keys1(id, k): up to two normalized 'title|artist' keys per row."""
    print(f"== normalizing {table}")
    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_tnorm AS
        SELECT s AS raw, nrm_key(s) AS k
        FROM (SELECT DISTINCT song AS s FROM {table} WHERE song IS NOT NULL)""")
    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_anorm AS
        SELECT a AS raw, nrm_key(a) AS k_full, nrm_main(a) AS k_main
        FROM (SELECT DISTINCT artist AS a FROM {table} WHERE artist IS NOT NULL)""")
    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_keys1 AS
        SELECT x.id, t.k || '|' || a.k_full AS k
        FROM {table} x JOIN {table}_tnorm t ON x.song = t.raw
                       JOIN {table}_anorm a ON x.artist = a.raw
        WHERE t.k <> '' AND a.k_full <> ''
        UNION
        SELECT x.id, t.k || '|' || a.k_main AS k
        FROM {table} x JOIN {table}_tnorm t ON x.song = t.raw
                       JOIN {table}_anorm a ON x.artist = a.raw
        WHERE t.k <> '' AND a.k_main <> ''""")
    c = con.execute(f"SELECT count(*), count(DISTINCT id) FROM {table}_keys1").fetchone()
    total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"  {table}_keys1: {c[0]:,} keys for {c[1]:,} / {total:,} rows "
          f"({total - c[1]:,} rows have no usable key)")


def build_matches(con):
    print("== matching")
    con.execute("""
        CREATE OR REPLACE TABLE matches1 AS
        SELECT p.base_id, p.lyr_id, abs(bs.year - ly.year) AS year_diff
        FROM (SELECT DISTINCT a.id AS base_id, b.id AS lyr_id
              FROM base_keys1 a JOIN lyr_keys1 b ON a.k = b.k) p
        JOIN base bs ON bs.id = p.base_id
        JOIN lyr ly ON ly.id = p.lyr_id""")
    c = con.execute("SELECT count(*) FROM matches1").fetchone()[0]
    print(f"  matches1: {c:,} (base row, lyr row) pairs")


def report(con, max_year_diff=2):
    """Counts of base rows with a lyr match, split into hits / non-hits. No rematching needed."""
    tot = con.execute("""SELECT count(*), count(*) FILTER (WHERE TRY_CAST(hit AS INTEGER) = 1)
                         FROM base""").fetchone()
    n_all, n_hit = tot
    n_non = n_all - n_hit
    print(f"base rows: {n_all:,} ({n_hit:,} hits, {n_non:,} non-hits)")
    conds = [("title + artist only", "TRUE"), ("+ same year", "m.year_diff = 0")]
    conds += [(f"+ year within +/-{k}", f"m.year_diff <= {k}") for k in range(1, max_year_diff + 1)]
    print(f"{'':<24}{'all':>18}{'hits':>18}{'non-hits':>18}")
    for label, cond in conds:
        a, h = con.execute(f"""
            SELECT count(DISTINCT m.base_id),
                   count(DISTINCT m.base_id) FILTER (WHERE TRY_CAST(b.hit AS INTEGER) = 1)
            FROM matches1 m JOIN base b ON b.id = m.base_id WHERE {cond}""").fetchone()
        cell = lambda x, d: f"{x:,} ({x / d:.1%})" if d else f"{x:,}"
        print(f"{label:<24}{cell(a, n_all):>18}{cell(h, n_hit):>18}{cell(a - h, n_non):>18}")
    s = con.execute(f"""
        SELECT median(c), max(c) FROM (SELECT count(*) c FROM matches1
                                       WHERE year_diff <= {max_year_diff} GROUP BY base_id)""").fetchone()
    if s[0] is not None:
        print(f"lyr rows per matched base row (within +/-{max_year_diff}): median {s[0]:.0f}, max {s[1]}")


def unmatched(con, limit=30, hits_only=True, year_tol=1):
    """Random sample of base rows with no lyr match, to see what the normalization misses."""
    where = "TRY_CAST(hit AS INTEGER) = 1 AND " if hits_only else ""
    return con.sql(f"""
        SELECT song, artist, year, hit FROM base
        WHERE {where} id NOT IN (SELECT base_id FROM matches1 WHERE year_diff <= {int(year_tol)})
        USING SAMPLE {int(limit)} ROWS""")


def run(report_years=2):
    con = n.connect(DB_PATH, memory_limit=MEMORY_LIMIT)
    register(con)
    step_load(con)
    build_keys(con, "base")
    build_keys(con, "lyr")
    build_matches(con)
    report(con, report_years)
    return con


if __name__ == "__main__":
    run().close()