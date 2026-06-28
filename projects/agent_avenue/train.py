"""Entraînement de l'agent Agent Avenue par évolution en self-play.

On fait évoluer un vecteur de poids (la fonction de valeur). À chaque génération
on crée des descendants en mutant les meilleurs parents, et on garde les
meilleurs selon leur taux de victoire contre un « pool » d'adversaires.

Important : grâce au lookahead (l'agent voit toujours les coups immédiatement
gagnants/perdants), même des poids quelconques jouent déjà correctement contre
l'aléatoire. Le vrai progrès se mesure donc en **head-to-head contre une
référence figée** : le champion de la génération 0. On suit ce taux de victoire
au fil des générations — c'est lui qui doit grimper (« gen 40 bat gen 0 »).

On sauvegarde des checkpoints à plusieurs générations : on les fera s'affronter
et on regardera la progression.

Usage :
    python train.py                       # 40 générations
    python train.py --generations 60
"""

import argparse
import os

import numpy as np

from policies import RandomPolicy, ValuePolicy, N_FEATURES
from match import run_game

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def winrate(w, opp_pol, n_games, seed0):
    """Taux de victoire d'une ValuePolicy(w) contre opp_pol, joué des deux côtés."""
    me = ValuePolicy(w)
    wins = 0
    for i in range(n_games):
        a, _, _ = run_game(me, opp_pol, seed=seed0 + i)
        wins += (a == 0)
        b, _, _ = run_game(opp_pol, me, seed=seed0 + 5000 + i)
        wins += (b == 1)
    return wins / (2 * n_games)


def milestones_for(generations):
    ms = {0, generations}
    for frac in (0.05, 0.15, 0.3, 0.5, 0.75):
        ms.add(int(round(generations * frac)))
    return sorted(ms)


def train(generations=40, mu=4, lam=8, n_games=12, track_games=60, sigma0=0.4, seed=0):
    rng = np.random.default_rng(seed)
    random_pol = RandomPolicy(seed=12345)
    os.makedirs(CKPT_DIR, exist_ok=True)
    milestones = milestones_for(generations)

    parents = [rng.normal(0, 0.3, N_FEATURES) for _ in range(mu)]
    baseline = None                    # champion figé de la génération 0
    champions = []
    history = []                       # (gen, winrate vs baseline, winrate vs random)

    for gen in range(generations + 1):
        sigma = sigma0 * (1 - 0.8 * gen / max(1, generations))
        pop = list(parents)
        while len(pop) < mu + lam:
            base = parents[rng.integers(len(parents))]
            pop.append(base + rng.normal(0, sigma, N_FEATURES))

        seed0 = int(rng.integers(1_000_000))
        scored = []
        for w in pop:
            wr_rand = winrate(w, random_pol, n_games, seed0)
            if baseline is None:
                fit = wr_rand
            else:
                wr_base = winrate(w, ValuePolicy(baseline), n_games, seed0 + 100)
                wr_champ = np.mean([winrate(w, ValuePolicy(c), n_games, seed0 + 200)
                                    for c in champions[-2:]])
                fit = 0.34 * wr_rand + 0.33 * wr_base + 0.33 * wr_champ
            scored.append((fit, w))
        scored.sort(key=lambda x: x[0], reverse=True)
        parents = [w for _, w in scored[:mu]]
        best_w = parents[0]

        if baseline is None:
            baseline = best_w.copy()
            np.save(os.path.join(CKPT_DIR, "aa_baseline.npy"), baseline)
        champions.append(best_w.copy())
        champions = champions[-4:]

        # métriques de suivi (plus de parties => courbe lisse)
        wr_base = winrate(best_w, ValuePolicy(baseline), track_games, seed0 + 777)
        wr_rand = winrate(best_w, random_pol, track_games, seed0 + 999)
        history.append((gen, wr_base, wr_rand))
        print(f"  gen {gen:>3}  | vs gen0 {wr_base*100:5.1f}%  "
              f"| vs random {wr_rand*100:5.1f}%  | sigma {sigma:.2f}")

        if gen in milestones:
            np.save(os.path.join(CKPT_DIR, f"aa_gen{gen}.npy"), best_w)

    np.save(os.path.join(CKPT_DIR, "aa_best.npy"), best_w)
    np.save(os.path.join(CKPT_DIR, "history.npy"), np.array(history))
    print("Entrainement termine. Meilleur vecteur :", np.round(best_w, 2))
    return best_w, history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entraine l'agent Agent Avenue (evolution).")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--games", type=int, default=12, help="parties/eval/cote")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(generations=args.generations, n_games=args.games, seed=args.seed)
