"""Fonctions partagées du projet hit-factors.

Tout ce qui doit être identique pour toute l'équipe (chemins, chargement des
sources, normalisation des titres/artistes pour la jointure) vit ici, pour que
chaque notebook n'ait pas sa propre version.
"""

from hit_factors.paths import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, ROOT
from hit_factors.text import join_key, normalize_artist, normalize_title, split_artists

__all__ = [
    "ROOT",
    "DATA_RAW",
    "DATA_INTERIM",
    "DATA_PROCESSED",
    "normalize_title",
    "normalize_artist",
    "split_artists",
    "join_key",
]
