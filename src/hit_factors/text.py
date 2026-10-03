"""Normalisation des titres et artistes, pour joindre les sources entre elles.

Billboard, Spotify, Genius et MusicBrainz n'ont pas d'identifiant commun : on
joint sur ``titre normalisé + artiste principal normalisé``, puis on rattrape
les cas restants par fuzzy matching (rapidfuzz).
"""

import re
import unicodedata

# Séparateurs de collaborations rencontrés dans Billboard / Spotify / Genius
_FEAT_RE = re.compile(r"\s*[\(\[]?\b(?:feat\.?|featuring|ft\.?|with)\s+", re.IGNORECASE)
# « x » minuscule seulement, pour ne pas couper « Lil Nas X » ; pas de « and »,
# pour ne pas couper « Florence and the Machine » ; pas de « + » (« Florence + The Machine »).
_SPLIT_RE = re.compile(
    r"\s*(?:,|&|/|(?<=\s)x(?=\s)|\b(?i:featuring|feat\.?|ft\.?|with)(?=\s))\s*"
)
# Mentions de version à retirer : "- Remastered 2011", "(Radio Edit)", "[Live]"…
_VERSION_RE = re.compile(
    r"\s*(?:-\s*|\(|\[)\s*(?:\d{4}\s+)?(?:remaster(?:ed)?|radio edit|single version|"
    r"album version|live|mono|stereo|explicit|clean|remix|edit|version)\b.*$",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _clean(text: str) -> str:
    text = _strip_accents(text.lower())
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: str) -> str:
    """'Blinding Lights - Remastered 2020' -> 'blinding lights'."""
    if not isinstance(title, str):
        return ""
    title = _FEAT_RE.split(title)[0]
    title = _VERSION_RE.sub("", title)
    return _clean(title)


def split_artists(artists: str) -> list[str]:
    """'Drake Featuring Rihanna & Future' -> ['Drake', 'Rihanna', 'Future'].

    Le premier élément est l'artiste principal. Attention : certains noms
    contiennent eux-mêmes un séparateur (« Simon & Garfunkel », « Earth, Wind &
    Fire ») ; ajoutez-les à une liste d'exceptions si la jointure en souffre.
    """
    if not isinstance(artists, str):
        return []
    return [a.strip() for a in _SPLIT_RE.split(artists) if a and a.strip()]


def normalize_artist(artists: str) -> str:
    """Artiste principal normalisé : 'Beyoncé Featuring JAY-Z' -> 'beyonce'."""
    parts = split_artists(artists)
    return _clean(parts[0]) if parts else ""


def join_key(title: str, artists: str) -> str:
    """Clé de jointure commune à toutes les sources : 'titre|artiste principal'."""
    return f"{normalize_title(title)}|{normalize_artist(artists)}"
