# hit-factors

**Qu'est-ce qui fait percer un tube ?** Analyse statistique et textuelle de milliers de morceaux pour identifier les facteurs qui font qu'une chanson entre au Billboard Hot 100.

Projet étudiant de data science, 9 personnes.

## Les 4 axes

| Notebook | Axe | Questions |
|---|---|---|
| [`00_cible_et_jointure`](notebooks/00_cible_et_jointure.ipynb) | Socle commun | Définir le succès, construire les négatifs, fusionner les sources dans `data/processed/morceaux.parquet`. |
| [`01_adn_du_morceau`](notebooks/01_adn_du_morceau.ipynb) | ADN du morceau | Durée, genre, caractéristiques audio, indépendant vs label. |
| [`02_paroles_nlp`](notebooks/02_paroles_nlp.ipynb) | Paroles (NLP) | Thèmes, sentiment, répétition, richesse du vocabulaire. |
| [`03_contexte_de_sortie`](notebooks/03_contexte_de_sortie.ipynb) | Contexte de sortie | Saison et jour de sortie, délai avant classement, featurings. |
| [`04_modelisation`](notebooks/04_modelisation.ipynb) | Modélisation | Prédire le succès et expliquer quels facteurs pèsent le plus. |

Le notebook `00` passe en premier : tous les autres lisent la table qu'il produit.

## Données

Billboard Hot 100 (dataset utdata), Kaggle Spotify Tracks, Kaggle Genius Song Lyrics, MusicBrainz. Voir [`data/README.md`](data/README.md) pour les télécharger.

## Installation

```bash
git clone https://github.com/adrienlefebvre6/hit-factors.git
cd hit-factors
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt  # installe aussi le dossier src/ comme paquet
cp .env.example .env             # puis remplir les clés
jupyter lab
```

## Organisation du dépôt

```
data/
  raw/          fichiers téléchargés, jamais modifiés (non versionnés)
  interim/      une table nettoyée par source
  processed/    table fusionnée morceaux.parquet
notebooks/      un notebook par axe
src/hit_factors/  fonctions partagées : chemins, chargement, normalisation titres/artistes
tests/          tests des fonctions partagées (pytest)
```

**Règle d'équipe :** une fonction utilisée par plus d'un notebook va dans `src/hit_factors/`, pas en copier-coller. En particulier, la jointure entre sources passe toujours par `hit_factors.join_key(titre, artistes)` et le comptage des featurings par `hit_factors.split_artists(artistes)`.

## Travailler à plusieurs

- Une branche par tâche (`git checkout -b axe2-sentiment`), puis une pull request vers `main`.
- Deux personnes qui modifient le même notebook en même temps créent des conflits difficiles à résoudre : se répartir les sections, ou travailler dans un notebook brouillon (`notebooks/brouillons/prenom_*.ipynb`) et reporter le résultat.
- Vider les sorties des notebooks avant de committer s'ils deviennent lourds (Kernel > Restart & Clear Outputs).
- Lancer `pytest` avant de pousser une modification dans `src/`.
