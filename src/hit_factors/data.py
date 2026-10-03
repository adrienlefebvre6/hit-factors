"""Chargement des sources brutes. Voir data/README.md pour les téléchargements."""

import pandas as pd

from hit_factors.paths import DATA_RAW

BILLBOARD_URL = (
    "https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/"
    "data-out/hot-100-current.csv"
)


def load_billboard(refresh: bool = False) -> pd.DataFrame:
    """Billboard Hot 100, une ligne par (semaine, titre), depuis 1958.

    Téléchargé une fois dans data/raw/, puis relu depuis le disque.
    """
    path = DATA_RAW / "billboard_hot100.csv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(BILLBOARD_URL).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["chart_week"])
