"""Agent plus intelligent : recherche en profondeur (minimax + alpha-bêta) avec
détermination Monte-Carlo de l'information cachée (PIMC).

Idée : le jeu a de l'information cachée (carte face cachée, main adverse). On
échantillonne plusieurs « mondes » concrets compatibles avec ce qu'on observe
(on tire une main adverse plausible et un paquet), puis dans chaque monde on
joue un minimax parfait-information à profondeur fixe, en utilisant la fonction
de valeur apprise comme évaluation des feuilles. On moyenne sur les mondes.

`depth` = nombre de tours anticipés. depth=1 ≈ l'agent d'origine (1 coup) ;
depth=2/3 anticipent les réponses de l'adversaire -> jeu nettement plus fort.
"""

import numpy as np

from cards import CARD_NAMES, NAME_TO_ID, N_TYPES
from env import BOARD_N, WIN_CB, LOSE_DD
from policies import TOTAL_BY_ID, value, BIG


def eval_leaf(env, w):
    """Évaluation d'un état, en « avantage pour le joueur 0 » (A0).

    Conservé pour compat ; utilise la valeur linéaire d'origine (8 features).
    """
    if env.winner is not None:
        return BIG if env.winner == 0 else -BIG
    cb = [int(env.counts[0][WIN_CB]), int(env.counts[1][WIN_CB])]
    dd = [int(env.counts[0][LOSE_DD]), int(env.counts[1][LOSE_DD])]
    return value(env.gap(), cb, dd, 0, w, None)


def _expand(remaining):
    pool = []
    for i in range(N_TYPES):
        pool.extend([CARD_NAMES[i]] * int(max(0, remaining[i])))
    return pool


def determinize(env, me, rng, up=None):
    """Monde concret compatible avec la vue de `me` : on rééchantillonne la main
    du joueur caché (et le paquet) à partir des cartes encore invisibles.
    `up`, si fourni, est une carte que le joueur caché vient de jouer (on la
    force dans sa main)."""
    c = env.clone()
    hidden = 1 - me                      # le joueur dont je ne vois pas la main
    remaining = TOTAL_BY_ID.copy()
    for n in c.hands[me]:
        remaining[NAME_TO_ID[n]] -= 1
    remaining -= c.counts[0] + c.counts[1]
    if up is not None:
        remaining[NAME_TO_ID[up]] -= 1

    pool = _expand(remaining)
    rng.shuffle(pool)
    hand_size = len(c.hands[hidden])
    new_hand = [up] if up is not None else []
    while len(new_hand) < hand_size and pool:
        new_hand.append(pool.pop())
    c.hands[hidden] = new_hand
    c.deck = pool
    return c


class SearchPolicy:
    name = "search"

    def __init__(self, w, depth=2, n_worlds=6, seed=0, leaf_value=None):
        self.w = np.asarray(w, float)
        self.depth = depth
        self.n_worlds = n_worlds
        self.rng = np.random.default_rng(seed)
        # leaf_value(gap, cb, dd) -> avantage joueur 0. Defaut : valeur d'origine.
        self.leaf_value = leaf_value

    def _leaf(self, env):
        if env.winner is not None:
            return BIG if env.winner == 0 else -BIG
        if self.leaf_value is None:
            return eval_leaf(env, self.w)
        cb = [int(env.counts[0][WIN_CB]), int(env.counts[1][WIN_CB])]
        dd = [int(env.counts[0][LOSE_DD]), int(env.counts[1][LOSE_DD])]
        return self.leaf_value(env.gap(), cb, dd)

    # ---- minimax parfait-information sur un monde déterminé ---------------
    def _search(self, env, depth, alpha, beta):
        """Renvoie A0 (avantage joueur 0). env.active = joueur qui doit JOUER."""
        if depth <= 0:
            return self._leaf(env)
        active = env.active
        maximizing = (active == 0)
        plays = env.legal_plays(active)
        if not plays:
            return self._leaf(env)

        best = -np.inf if maximizing else np.inf
        for up, down in plays:
            v = self._recruit_value(env, up, down, depth, alpha, beta)
            if maximizing:
                best = max(best, v); alpha = max(alpha, best)
            else:
                best = min(best, v); beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def _recruit_value(self, env, up, down, depth, alpha, beta):
        """L'adversaire (non-actif) recrute pour optimiser SON camp."""
        active = env.active
        recruiter = 1 - active
        maximizing = (recruiter == 0)
        best = -np.inf if maximizing else np.inf
        for recruit_up in (True, False):
            c = env.clone()
            c.apply_turn(up, down, recruit_up)
            w = c.check_end()
            if w is not None:
                v = BIG if w == 0 else -BIG
            else:
                c.active = recruiter             # à l'adversaire de jouer
                v = self._search(c, depth - 1, alpha, beta)
            if maximizing:
                best = max(best, v); alpha = max(alpha, best)
            else:
                best = min(best, v); beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    # ---- décisions (moyennées sur plusieurs mondes) ----------------------
    def choose_play(self, env):
        me = env.active
        sign = 1.0 if me == 0 else -1.0       # « pour moi » = A0 * sign
        plays = env.legal_plays(me)
        worlds = [determinize(env, me, self.rng) for _ in range(self.n_worlds)]

        scores = np.zeros(len(plays))
        for w_env in worlds:
            for k, (up, down) in enumerate(plays):
                a0 = self._recruit_value(w_env, up, down, self.depth, -np.inf, np.inf)
                scores[k] += sign * a0
        best_up, best_down = plays[int(np.argmax(scores))]
        # ordre face visible/cachée : on cache la carte qu'on préfère garder
        return self._order(env, me, best_up, best_down)

    def _order(self, env, me, a, b):
        """Choisit laquelle des deux cartes mettre face visible (heuristique de
        bluff) : on rend visible celle qu'on préfère voir l'adversaire prendre."""
        sign = 1.0 if me == 0 else -1.0
        # valeur pour moi si JE recrute la carte x (l'adversaire prend l'autre)
        def mine(x, other):
            c = env.clone()
            c.apply_turn(x, other, recruit_up=False)   # adversaire prend 'other'
            c.check_end()
            return sign * self._leaf(c)
        va = sign * 0 + mine(a, b)     # je garde a
        vb = mine(b, a)                # je garde b
        # je veux GARDER la meilleure -> la mettre face cachée (down) ;
        # donc face visible = la moins bonne pour moi.
        if va >= vb:
            return (b, a)              # visible=b, je garde a (cachee)
        return (a, b)

    def choose_recruit(self, env, up):
        me = 1 - env.active
        active = env.active
        sign = 1.0 if me == 0 else -1.0
        tot_up, tot_down = 0.0, 0.0
        for _ in range(self.n_worlds):
            c = determinize(env, me, self.rng, up=up)
            downs = [n for n in c.hands[active] if n != up]
            down = downs[self.rng.integers(len(downs))] if downs else up
            tot_up += sign * self._eval_recruit(c, up, down, recruit_up=True)
            tot_down += sign * self._eval_recruit(c, up, down, recruit_up=False)
        return tot_up >= tot_down

    def _eval_recruit(self, env, up, down, recruit_up):
        c = env.clone()
        c.apply_turn(up, down, recruit_up)
        w = c.check_end()
        if w is not None:
            return BIG if w == 0 else -BIG
        c.active = 1 - env.active             # à l'adversaire (l'actif d'origine)
        return self._search(c, self.depth - 1, -np.inf, np.inf)
