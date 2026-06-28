"""Politiques de décision pour Agent Avenue.

Une politique doit savoir faire deux choses :
  - choose_play(env)   : quand c'est son tour, choisir (carte_visible, carte_cachée).
  - choose_recruit(env, up) : quand l'adversaire a joué, choisir de prendre la
                              carte visible `up` (True) ou la cachée (False).

L'agent qui apprend (`ValuePolicy`) évalue chaque coup en simulant son résultat
avec le moteur et en notant l'état obtenu via une fonction de valeur linéaire
`V = w · features`. L'information cachée (carte face cachée, main adverse) est
gérée en moyennant sur les cartes encore invisibles.
"""

import numpy as np

from cards import (CARD_DEFS, CARD_NAMES, NAME_TO_ID, N_TYPES, move_for)
from env import BOARD_N, START_GAP, WIN_CB, LOSE_DD

BIG = 1e6
N_FEATURES = 8

# nombre total d'exemplaires de chaque type, indexé par id
TOTAL_BY_ID = np.zeros(N_TYPES, dtype=int)
for _name, (_tbl, _qty) in CARD_DEFS.items():
    TOTAL_BY_ID[NAME_TO_ID[_name]] = _qty


# --------------------------------------------------------------------------
# Résolution hypothétique d'un tour (sans toucher au moteur réel)
# --------------------------------------------------------------------------
def resolve(gap, counts, active, card_active, card_opp):
    """Applique un tour hypothétique sans copier de tableaux.

    Renvoie (new_gap, cb, dd, winner) où cb/dd sont les compteurs Codebreaker /
    Daredevil [joueur0, joueur1] après le tour. C'est tout ce dont la fonction
    de valeur a besoin -> chemin chaud très léger pour l'entraînement.
    """
    opp = 1 - active
    ia, io = NAME_TO_ID[card_active], NAME_TO_ID[card_opp]
    m_a = move_for(card_active, int(counts[active][ia]) + 1)
    m_o = move_for(card_opp, int(counts[opp][io]) + 1)
    d0 = m_a if active == 0 else m_o
    d1 = m_a if active == 1 else m_o
    new_gap = gap + d1 - d0

    cb = [int(counts[0][WIN_CB]), int(counts[1][WIN_CB])]
    dd = [int(counts[0][LOSE_DD]), int(counts[1][LOSE_DD])]
    if ia == WIN_CB:
        cb[active] += 1
    if io == WIN_CB:
        cb[opp] += 1
    if ia == LOSE_DD:
        dd[active] += 1
    if io == LOSE_DD:
        dd[opp] += 1

    catch0, catch1 = new_gap <= 0, new_gap >= BOARD_N
    good0 = catch0 or cb[0] >= 3 or dd[1] >= 3
    good1 = catch1 or cb[1] >= 3 or dd[0] >= 3
    if good0 or good1:
        winner = active if (good0 and good1) else (0 if good0 else 1)
    else:
        winner = None
    return new_gap, cb, dd, winner


def value(new_gap, cb, dd, me, w, winner):
    """Valeur de l'état (du point de vue de `me`) à partir de scalaires."""
    if winner is not None:
        return BIG if winner == me else -BIG
    opp = 1 - me
    my_c = new_gap if me == 0 else BOARD_N - new_gap
    op_c = new_gap if opp == 0 else BOARD_N - new_gap
    return (w[0] * (op_c - my_c) / BOARD_N
            + w[1] * min(cb[me], 3) / 3
            - w[2] * min(cb[opp], 3) / 3
            - w[3] * min(dd[me], 3) / 3
            + w[4] * min(dd[opp], 3) / 3
            + w[5] * (1.0 if cb[me] == 2 else 0.0)
            + w[6] * (1.0 if cb[opp] == 2 else 0.0)
            + w[7])


# --------------------------------------------------------------------------
# Cartes encore invisibles pour `me` (pour gérer l'information cachée)
# --------------------------------------------------------------------------
def _unseen_names(env, me, exclude_name=None):
    remaining = TOTAL_BY_ID.copy()
    for n in env.hands[me]:
        remaining[NAME_TO_ID[n]] -= 1
    remaining -= env.counts[0] + env.counts[1]
    if exclude_name is not None:
        remaining[NAME_TO_ID[exclude_name]] -= 1
    names, probs = [], []
    total = remaining.sum()
    if total <= 0:
        return [], []
    for i in range(N_TYPES):
        if remaining[i] > 0:
            names.append(CARD_NAMES[i])
            probs.append(remaining[i] / total)
    return names, probs


# --------------------------------------------------------------------------
# Politiques
# --------------------------------------------------------------------------
class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def choose_play(self, env):
        plays = env.legal_plays(env.active)
        return plays[self.rng.integers(len(plays))]

    def choose_recruit(self, env, up):
        return bool(self.rng.integers(2))


class ValuePolicy:
    """Agent à fonction de valeur linéaire, lookahead d'un coup."""
    name = "value"

    def __init__(self, w=None, seed=0):
        self.w = np.zeros(N_FEATURES) if w is None else np.asarray(w, float)
        self.rng = np.random.default_rng(seed)

    # --- valeur espérée d'une option où une carte est inconnue ----------
    def _expected_value(self, env, me, active, known_is_active, known_card):
        """E[V(me)] en moyennant la carte inconnue sur les cartes invisibles."""
        gap, counts = env.gap(), env.counts
        names, probs = _unseen_names(env, me, exclude_name=known_card)
        if not names:                       # plus rien d'inconnu : carte neutre
            names, probs = [known_card], [1.0]
        total = 0.0
        for n, p in zip(names, probs):
            if known_is_active:
                ca, co = known_card, n
            else:
                ca, co = n, known_card
            g2, cb, dd, win = resolve(gap, counts, active, ca, co)
            total += p * value(g2, cb, dd, me, self.w, win)
        return total

    def choose_recruit(self, env, up):
        """`me` recrute. env.active est l'adversaire qui a joué up + une cachée."""
        me = 1 - env.active
        active = env.active
        # prendre up : ma carte = up (connue), l'actif prend la cachée (inconnue)
        v_up = self._expected_value(env, me, active, known_is_active=False, known_card=up)
        # prendre la cachée : ma carte inconnue, l'actif prend up (connue)
        v_down = self._expected_value(env, me, active, known_is_active=True, known_card=up)
        return v_up >= v_down

    def _opp_takes_up(self, env, me, up, down):
        """Modèle de l'adversaire : prend-il la carte visible ? Il ne voit que
        up ; il traite la carte cachée comme une carte « moyenne » invisible."""
        opp = me                          # ici on raisonne pour l'adversaire
        active = env.active               # = moi (l'actif)
        # NB: on réutilise _expected_value du point de vue de l'adversaire (1-active)
        opp_view = 1 - active
        v_up = self._expected_value(env, opp_view, active, known_is_active=True, known_card=up)
        v_down = self._expected_value(env, opp_view, active, known_is_active=False, known_card=up)
        return v_up >= v_down

    def choose_play(self, env):
        me = env.active
        gap, counts = env.gap(), env.counts
        best, best_v = None, -np.inf
        for up, down in env.legal_plays(me):
            opp_takes_up = self._opp_takes_up(env, me, up, down)
            if opp_takes_up:              # l'adversaire prend up, je prends down
                ca, co = down, up
            else:                         # il prend la cachée, je prends up
                ca, co = up, down
            g2, cb, dd, win = resolve(gap, counts, me, ca, co)
            v = value(g2, cb, dd, me, self.w, win)
            if v > best_v:
                best_v, best = v, (up, down)
        return best
