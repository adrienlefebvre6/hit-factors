"""Chemins du projet, valables quel que soit le dossier d'où l'on lance le notebook."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"  # fichiers tels que téléchargés, jamais modifiés
DATA_INTERIM = ROOT / "data" / "interim"  # nettoyés, une table par source
DATA_PROCESSED = ROOT / "data" / "processed"  # table fusionnée prête pour l'analyse
