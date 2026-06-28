"""Fonction de valeur « enrichie » (v2), apprise par expert-iteration.

Par rapport à la valeur linéaire d'origine (8 features), on ajoute des signaux
non linéaires et de « danger » : avance dans la course au carré, indicateurs
« à une carte de gagner / de perdre », etc. Tout reste calculable à partir de
l'écart et des compteurs Codebreaker/Daredevil, donc utilisable aussi bien dans
le chemin rapide que dans les feuilles de la recherche.

Les poids sont appris (voir train_value.py) pour prédire qui gagne la partie,
à partir de données générées par l'agent de recherche (l'« expert »).
"""

import numpy as np

from env import BOARD_N

N_FEATURES2 = 11


def feats2(gap, cb, dd, me):
    """Vecteur de features (point de vue de `me`) à partir de scalaires."""
    opp = 1 - me
    my_c = gap if me == 0 else BOARD_N - gap          # ma distance pour rattraper
    op_c = gap if opp == 0 else BOARD_N - gap
    race = (op_c - my_c) / BOARD_N
    mcb, ocb = min(cb[me], 3) / 3, min(cb[opp], 3) / 3
    mdd, odd = min(dd[me], 3) / 3, min(dd[opp], 3) / 3
    return np.array([
        race,                                   # 0 avance dans la course
        race * abs(race),                       # 1 accentue les grosses avances
        mcb,                                    # 2 ma progression codebreaker
        ocb,                                    # 3 celle de l'adversaire
        mdd,                                    # 4 mon danger daredevil
        odd,                                    # 5 le sien (bon pour moi)
        1.0 if cb[me] == 2 else 0.0,            # 6 à 1 carte de gagner (CB)
        1.0 if cb[opp] == 2 else 0.0,           # 7 adversaire à 1 carte (CB)
        1.0 if dd[me] == 2 else 0.0,            # 8 à 1 carte de perdre (DD)
        1.0 if dd[opp] == 2 else 0.0,           # 9 adversaire à 1 carte de perdre
        1.0,                                    # 10 biais
    ])


def make_leaf_value(w2):
    """Renvoie une fonction leaf_value(gap, cb, dd) -> score (avantage joueur 0)."""
    w2 = np.asarray(w2, float)

    def leaf(gap, cb, dd):
        return float(feats2(gap, cb, dd, 0) @ w2)
    return leaf
