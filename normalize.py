"""
Artist-name normalization for metadata matching.

Derived from code by Thierry Bertin-Mahieux (2011, Columbia University,
Million Song Dataset project), released under GPL v3.
Changes: Python 2 -> 3, raw-string regexes, removed the Levenshtein-based
comparison functions (only normalization is kept), empty variants are dropped,
results are cached per unique string, and pandas helpers were added.

Also contains the DuckDB pipeline (load csv -> normalize -> match count).

Each normalized artist is a frozenset of variants (lowercase, no spaces or
punctuation, with permutations of "A and B"), so two artist strings can later
be compared with a set intersection.
"""
import ast
import itertools
import re
import unicodedata
from functools import lru_cache

import duckdb
import pandas as pd

# ROTATION SYMBOLS (A and B => B and A)
rotation_symbols = [r'\|', '/', '&', ',', r'\+', ';', '_']
rotation_words = ['and', 'y', 'et', 'vs', 'vs.', 'v', 'with', 'feat',
                  'feat.', 'featuring', 'presents', 'ft.', 'pres.']

# SYMBOLS TO REMOVE AT THE BEGINNING
stub_to_remove = ['dj', 'dj.', 'mc', 'm.c.', 'mc.', 'the', 'los', 'les']

# SYMBOLS TO REMOVE AT THE END
end_to_remove1 = ['big band', 'trio', 'quartet', 'ensemble', 'orchestra']
end_to_remove2 = ['band']

# COMPILED REGULAR EXPRESSIONS
re_space = re.compile(r'\s')
re_nonalphanum = re.compile(r'\W')
re_rotsymbols = re.compile(r'\s*(?:' + '|'.join(rotation_symbols) + r')\s*')
re_rotwords = re.compile(
    r'\s(' + '|'.join(re.escape(w) for w in rotation_words) + r')\s')
re_remstub = re.compile(
    '(' + '|'.join(re.escape(w) for w in stub_to_remove) + r')\s(.*)')
re_remending1 = re.compile(r'(.*)\s(' + '|'.join(end_to_remove1) + ')')
re_remending2 = re.compile(r'(.*)\s(' + '|'.join(end_to_remove2) + ')')
re_remquotes = re.compile(r'(.+)\s(".+?")\s(.+)')
re_remparenthesis = re.compile(r'(.+)\s(\(.+?\))\s*(.*)')


def remove_non_ascii(s):
    """Decompose accents (e -> e + accent mark), then drop non-ASCII chars."""
    decomposition = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in decomposition if c < '\x7f')


def remove_spaces(s):
    return re_space.sub('', s)


def replace_rotation_symbols(s):
    """Replace '&', '/', ',', ... by ' and '."""
    return re_rotsymbols.sub(' and ', s)


def remove_stub(s):
    """Remove a questionable beginning, e.g. 'dj ', 'the '."""
    m = re_remstub.match(s)
    return m.group(2) if m else s


def remove_endings(s):
    """Remove questionable endings, e.g. ' band', ' orchestra'."""
    m = re_remending1.match(s)
    if m:
        s = m.group(1)
    m = re_remending2.match(s)
    if m:
        s = m.group(1)
    return s


def remove_quotes(s):
    """Thierry "The Awesomest" BM -> Thierry BM"""
    m = re_remquotes.match(s)
    return m.group(1) + ' ' + m.group(3) if m else s


def remove_parenthesis(s):
    """Thierry (Coolest guy) -> Thierry"""
    m = re_remparenthesis.match(s)
    if not m:
        return s
    return m.group(1) + ' ' + m.group(3) if m.group(3) else m.group(1)


def normalize_no_rotation(s):
    """Normalize a name that is supposed to contain no rotation term."""
    s = remove_stub(s)
    s = remove_endings(s)
    s = remove_parenthesis(s)
    s = remove_quotes(s)
    return s


def split_rotation_words(s):
    """
    Split on 'and', 'vs', 'featuring', ... then build every permutation of
    the parts (so "a and b" and "b and a" share a variant).
    """
    parts = [p for p in re_rotwords.split(s) if p not in rotation_words][:5]
    results = set(parts)
    for perm in itertools.permutations(parts):
        results.add(' '.join(perm))
    # redo the same after stripping stubs/endings from each part
    parts = [normalize_no_rotation(p) for p in parts]
    results.update(parts)
    for perm in itertools.permutations(parts):
        results.add(' '.join(perm))
    return results


def remove_nonalphanumeric(s):
    """Replace punctuation by spaces and collapse runs of whitespace."""
    return ' '.join(p for p in re_nonalphanum.split(s) if p)


def normalize_artist(s):
    """Return the set of normalized variants of one artist string."""
    results = set()
    s = s.lower()
    results.add(s)                      # keeps non-Latin scripts intact
    s = remove_non_ascii(s)
    results.add(s)
    results.add(remove_parenthesis(s))  # in case there is an '&' in the ()
    s = replace_rotation_symbols(s)
    results.update(split_rotation_words(s))
    results = {remove_spaces(remove_nonalphanumeric(r)) for r in results}
    results.discard('')                 # empty strings must never match
    return results


# ---------------------------------------------------------------------------
# pandas helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _norm_one(s):
    return frozenset(normalize_artist(s))


def parse_artist_list(x):
    """"['A', 'B']" (a string holding a list) -> ['A', 'B']"""
    try:
        v = ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return [x]
    if isinstance(v, (list, tuple)):
        return [str(a) for a in v]
    return [str(v)]


@lru_cache(maxsize=None)
def _norm_spotify_str(x):
    out = set()
    for a in parse_artist_list(x):
        out |= _norm_one(a)
    return frozenset(out)


def norm_billboard(x):
    """Billboard-style string: 'Wizkid Featuring Justin Bieber & Tems'."""
    return _norm_one(x) if isinstance(x, str) else frozenset()


def norm_spotify(x):
    """Spotify-style string holding a list: "['ABS', 'Nodesha']"."""
    return _norm_spotify_str(x) if isinstance(x, str) else frozenset()


# ===========================================================================
# DuckDB pipeline: load csv -> normalize artists -> count matches
# Everything below runs inside DuckDB; only the DISTINCT artist strings are
# pulled into Python (to run the normalizer), never the full tables.
#
# Standard column names used in the tables (map your csv headers to these in
# load_csv):   df     : song, artists, year
#              df_hot : song, artists            (year optional)
#              lyr    : song, artist, features, year
# ===========================================================================
_EMPTY = frozenset()


def _clean_features(x):
    """'{"Drake / Kanye West / Lil Wayne & Eminem"}' -> plain text; '{}' -> ''"""
    if not isinstance(x, str):
        return ""
    return x.replace('"', "").strip().strip("{}").strip()


def norm_features(x):
    s = _clean_features(x)
    return norm_billboard(s) if s else _EMPTY


def connect(db_path="songs.duckdb", memory_limit="4GB", threads=None,
            temp_dir="duckdb_tmp"):
    """Open (or create) a database file. Big joins spill to temp_dir instead of crashing."""
    con = duckdb.connect(db_path)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    return con


def load_csv(con, table, path, columns, read_options="ignore_errors=true"):
    """
    Load only the needed columns of a csv into a DuckDB table.
    columns: {csv_header: standard_name}, e.g. {"name": "song", "artists": "artists", "year": "year"}
             (or a plain list if the headers already are the standard names).
    Adds `id` (row number) and `title_key` (lowercased, trimmed song).
    """
    if not isinstance(columns, dict):
        columns = {c: c for c in columns}
    if "song" not in columns.values():
        raise ValueError("columns must map one csv column to 'song'")
    sel = []
    for raw, std in columns.items():
        col = '"' + raw.replace('"', '""') + '"'
        if std == "year":
            col = f"TRY_CAST(TRY_CAST({col} AS DOUBLE) AS INTEGER)"
        sel.append(f"{col} AS {std}")
    p = str(path).replace("'", "''")
    opts = f", {read_options}" if read_options else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT row_number() OVER () AS id, *, lower(trim(song)) AS title_key
        FROM (SELECT {', '.join(sel)}
              FROM read_csv('{p}', header=true, all_varchar=true{opts}))
    """)
    n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n:,} rows loaded")


def _table_exists(con, name):
    return con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                       [name]).fetchone()[0] > 0


def _columns(con, table):
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def _build_lookup(con, table, col, lookup, norm_fn, batch=50_000, reuse=False):
    """lookup(raw, variant): one row per normalized variant of each DISTINCT raw string.
    reuse=True skips the slow Python step if the lookup table already exists."""
    if reuse and _table_exists(con, lookup):
        print(f"  {lookup} already exists, reusing it")
        return
    con.execute(f"CREATE OR REPLACE TABLE {lookup} (raw VARCHAR, variant VARCHAR)")
    raws = [r[0] for r in con.execute(
        f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL").fetchall()]
    print(f"  normalizing {len(raws):,} distinct strings of {table}.{col}")
    for i in range(0, len(raws), batch):
        rows = [(r, v) for r in raws[i:i + batch] for v in norm_fn(r)]
        if rows:
            con.register("_batch", pd.DataFrame(rows, columns=["raw", "variant"]))
            con.execute(f"INSERT INTO {lookup} SELECT raw, variant FROM _batch")
            con.unregister("_batch")
    _norm_one.cache_clear()
    _norm_spotify_str.cache_clear()


def _build_variants(con, table, lookups):
    """{table}_v(id, title_key, variant): the normalized artist variants of every row."""
    parts = [f"SELECT t.id, t.title_key, l.variant FROM {table} t "
             f"JOIN {lk} l ON t.{col} = l.raw" for col, lk in lookups]
    con.execute(f"CREATE OR REPLACE TABLE {table}_v AS " + " UNION ALL ".join(parts))
    n = con.execute(f"SELECT count(*) FROM {table}_v").fetchone()[0]
    print(f"  {table}_v: {n:,} (row, variant) pairs")


def normalize_df(con, table="df", artists_col="artists", reuse=False):
    """1mil2: artists are list strings like "['A', 'B']"."""
    _build_lookup(con, table, artists_col, f"{table}_lookup", norm_spotify, reuse=reuse)
    _build_variants(con, table, [(artists_col, f"{table}_lookup")])


def normalize_hot(con, table="df_hot", artists_col="artists", reuse=False):
    """Billboard: plain strings like 'A Featuring B & C'."""
    _build_lookup(con, table, artists_col, f"{table}_lookup", norm_billboard, reuse=reuse)
    _build_variants(con, table, [(artists_col, f"{table}_lookup")])


def normalize_lyr(con, table="lyr", artist_col="artist", features_col="features", reuse=False):
    """Lyrics: one main artist column + one features column ({} or {"A / B"})."""
    _build_lookup(con, table, artist_col, f"{table}_lookup", norm_billboard, reuse=reuse)
    _build_lookup(con, table, features_col, f"{table}_feat_lookup", norm_features, reuse=reuse)
    _build_variants(con, table, [(artist_col, f"{table}_lookup"),
                                 (features_col, f"{table}_feat_lookup")])


def match_report(con, left, matches, year_tol=1, use_year=True, left_artist_col="artists"):
    """Print how many `left` rows have a match. Re-runnable with another year_tol, no rematching."""
    n = con.execute(f"SELECT count(*) FROM {left}").fetchone()[0]

    def rows(cond="TRUE"):
        return con.execute(
            f"SELECT count(DISTINCT left_id) FROM {matches} WHERE {cond}").fetchone()[0]

    ok = f"year_diff <= {int(year_tol)}" if use_year else "TRUE"
    key = f"title_key || '||' || coalesce({left_artist_col}, '')"
    n_songs = con.execute(f"SELECT count(DISTINCT {key}) FROM {left}").fetchone()[0]
    m_songs = con.execute(f"""
        SELECT count(DISTINCT {key}) FROM {left}
        WHERE id IN (SELECT left_id FROM {matches} WHERE {ok})""").fetchone()[0]
    stats = con.execute(f"""
        SELECT median(c), max(c) FROM
          (SELECT count(*) c FROM {matches} WHERE {ok} GROUP BY left_id)""").fetchone()

    any_m = rows()
    print(f"{left} rows: {n:,}")
    print(f"  title + artist only      : {any_m:,} ({any_m / n:.1%})")
    if use_year:
        ex = rows("year_diff = 0")
        print(f"  + same year              : {ex:,} ({ex / n:.1%})")
    wi = rows(ok)
    label = f"+ year within +/-{year_tol}" if use_year else "(final)"
    print(f"  {label:<25}: {wi:,} ({wi / n:.1%})")
    print(f"  unique songs matched     : {m_songs:,} / {n_songs:,} ({m_songs / n_songs:.1%})")
    if stats[0] is not None:
        print(f"  right rows per matched left row: median {stats[0]:.0f}, max {stats[1]}")
    return {"rows": n, "title_artist": any_m, "final": wi, "songs": m_songs}


def match_count(con, left, right, matches, use_year=True, year_tol=1,
                left_artist_col="artists"):
    """
    Build `matches`(left_id, right_id, year_diff) = rows with the same title_key
    and at least one shared artist variant, then print the report.
    Needs {left}_v and {right}_v (run the normalize_* functions first).
    """
    year = "abs(l.year - r.year)" if use_year else "CAST(NULL AS INTEGER)"
    con.execute(f"""
        CREATE OR REPLACE TABLE {matches} AS
        SELECT p.left_id, p.right_id, {year} AS year_diff
        FROM (SELECT DISTINCT a.id AS left_id, b.id AS right_id
              FROM {left}_v a JOIN {right}_v b
                ON a.title_key = b.title_key AND a.variant = b.variant) p
        JOIN {left} l ON l.id = p.left_id
        JOIN {right} r ON r.id = p.right_id
    """)
    return match_report(con, left, matches, year_tol, use_year, left_artist_col)


# ---------------------------------------------------------------------------
# ISRC validation of the title + artist matches
# ---------------------------------------------------------------------------
def isrc_check(con, left, right, matches, isrc_col="isrc"):
    """
    Compare the title+artist matches in `matches` against ISRC (needs an isrc
    column in both tables: add "isrc": "isrc" in load_csv).

    1) Precision: of the title+artist pairs, how many share the same ISRC?
       Pairs where an ISRC is missing on either side can't be checked.
    2) Recall: of the pairs that share an ISRC, how many did title+artist find?

    Creates:
      {matches}_mismatch : title+artist pairs whose ISRCs differ (likely false matches
                           or different versions of a song)
      {matches}_isrc     : all pairs that share an ISRC (left_id, right_id)
    """
    for t in (left, right):
        if isrc_col not in _columns(con, t):
            raise ValueError(f"{t} has no '{isrc_col}' column; reload it with "
                             f'load_csv(..., {{..., "<csv isrc header>": "{isrc_col}"}})')

    def clean(alias):
        return f"nullif(upper(replace(trim({alias}.{isrc_col}), '-', '')), '')"

    con.execute(f"""
        CREATE OR REPLACE TABLE {matches}_cmp AS
        SELECT m.left_id, m.right_id, {clean('l')} AS left_isrc, {clean('r')} AS right_isrc
        FROM {matches} m
        JOIN {left} l ON l.id = m.left_id
        JOIN {right} r ON r.id = m.right_id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {matches}_mismatch AS
        SELECT * FROM {matches}_cmp
        WHERE left_isrc IS NOT NULL AND right_isrc IS NOT NULL AND left_isrc <> right_isrc
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {matches}_isrc AS
        SELECT DISTINCT l.id AS left_id, r.id AS right_id
        FROM {left} l JOIN {right} r ON {clean('l')} = {clean('r')}
    """)

    both = "left_isrc IS NOT NULL AND right_isrc IS NOT NULL"
    pr = con.execute(f"""
        SELECT count(*),
               count(*) FILTER (WHERE {both}),
               count(*) FILTER (WHERE left_isrc = right_isrc),
               count(DISTINCT left_id),
               count(DISTINCT left_id) FILTER (WHERE {both}),
               count(DISTINCT left_id) FILTER (WHERE left_isrc = right_isrc)
        FROM {matches}_cmp""").fetchone()
    pairs, chk, same, rows, rows_chk, rows_ok = pr

    def pct(a, b):
        return f"{a / b:.1%}" if b else "n/a"

    print(f"Title+artist matches: {pairs:,} pairs, {rows:,} {left} rows")
    print("Precision (does ISRC agree?)")
    print(f"  pairs where both sides have an ISRC : {chk:,} ({pct(chk, pairs)})")
    print(f"  ...of those, same ISRC              : {same:,} ({pct(same, chk)})")
    print(f"  ...of those, different ISRC         : {chk - same:,} ({pct(chk - same, chk)})  -> {matches}_mismatch")
    print(f"  {left} rows checkable               : {rows_chk:,} ({pct(rows_chk, rows)})")
    print(f"  ...with at least one same-ISRC pair : {rows_ok:,} ({pct(rows_ok, rows_chk)})")

    rc = con.execute(f"""
        SELECT (SELECT count(*) FROM {matches}_isrc),
               (SELECT count(*) FROM {matches}_isrc i
                  WHERE EXISTS (SELECT 1 FROM {matches} m
                                WHERE m.left_id = i.left_id AND m.right_id = i.right_id)),
               (SELECT count(DISTINCT left_id) FROM {matches}_isrc),
               (SELECT count(DISTINCT left_id) FROM {matches}_isrc
                  WHERE left_id IN (SELECT left_id FROM {matches}))
    """).fetchone()
    ipairs, ifound, irows, irows_found = rc
    print("Recall (does title+artist find the ISRC matches?)")
    print(f"  pairs sharing an ISRC               : {ipairs:,}")
    print(f"  ...also found by title+artist       : {ifound:,} ({pct(ifound, ipairs)})")
    print(f"  {left} rows with an ISRC match      : {irows:,}")
    print(f"  ...also found by title+artist       : {irows_found:,} ({pct(irows_found, irows)})  "
          f"(missed ones: {matches}_isrc pairs not in {matches})")
    return {"pairs": pairs, "checkable": chk, "same": same,
            "isrc_pairs": ipairs, "isrc_pairs_found": ifound}