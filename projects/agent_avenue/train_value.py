"""Expert-iteration : on apprend une meilleure fonction de valeur (v2).

Principe (façon AlphaZero, en version minuscule) :
  1. un « expert » fort = l'agent de recherche (minimax + PIMC) joue des parties
     contre lui-même ;
  2. on enregistre les états traversés et qui gagne au final ;
  3. on ajuste la valeur enrichie (régression logistique) pour prédire l'issue ;
  4. on recommence en utilisant la nouvelle valeur dans la recherche (l'expert
     devient meilleur) — une ou deux itérations suffisent.

Une meilleure évaluation des feuilles rend la recherche plus forte à TOUTES les
profondeurs : c'est un modèle plus intelligent, pas seulement plus de calcul.

Usage :
    python train_value.py                 # 2 itérations
"""

import argparse
import os

import numpy as np

from env import AgentAvenue, WIN_CB, LOSE_DD
from search import SearchPolicy
from smart_value import feats2, make_leaf_value, N_FEATURES2

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def play_and_record(polA, polB, seed, max_turns=120):
    """Joue une partie et renvoie (états, winner). États = (gap, cb, dd) après
    chaque tour, hors états terminaux."""
    env = AgentAvenue(seed=seed)
    pols = [polA, polB]
    states = []
    winner = None
    for _ in range(max_turns):
        active, opp = env.active, 1 - env.active
        if len(env.hands[active]) < 2:
            d0, d1 = env.catch_distance(0), env.catch_distance(1)
            winner = 0 if d0 < d1 else (1 if d1 < d0 else active)
            env.winner = winner
            break
        up, down = pols[active].choose_play(env)
        recruit_up = pols[opp].choose_recruit(env, up)
        env.apply_turn(up, down, recruit_up)
        winner = env.check_end()
        if winner is not None:
            break
        cb = [int(env.counts[0][WIN_CB]), int(env.counts[1][WIN_CB])]
        dd = [int(env.counts[0][LOSE_DD]), int(env.counts[1][LOSE_DD])]
        states.append((env.gap(), cb, dd))
        env.active = opp
    return states, winner


def build_dataset(expert, n_games, seed0):
    X, y = [], []
    for g in range(n_games):
        states, winner = play_and_record(expert, expert, seed=seed0 + g)
        if winner is None:
            continue
        for gap, cb, dd in states:
            # deux exemples (un par point de vue) -> donnees doublees
            X.append(feats2(gap, cb, dd, 0)); y.append(1.0 if winner == 0 else 0.0)
            X.append(feats2(gap, cb, dd, 1)); y.append(1.0 if winner == 1 else 0.0)
    return np.array(X), np.array(y)


def fit_logistic(X, y, epochs=400, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(epochs):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / n + l2 * w
        w -= lr * grad
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    acc = float(np.mean((p > 0.5) == (y > 0.5)))
    return w, acc


def evaluate(leaf_a, leaf_b, w, depth=2, n=40):
    """Taux de victoire de la recherche(leaf_a) contre la recherche(leaf_b)."""
    from match import run_game
    pa = SearchPolicy(w, depth=depth, n_worlds=4, leaf_value=leaf_a)
    pb = SearchPolicy(w, depth=depth, n_worlds=4, leaf_value=leaf_b)
    wa = 0
    for i in range(n):
        r, _, _ = run_game(pa, pb, seed=i); wa += (r == 0)
        r, _, _ = run_game(pb, pa, seed=5000 + i); wa += (r == 1)
    return wa / (2 * n)


def train(iterations=2, games=400, depth=2, seed=0):
    w_lin = np.load(os.path.join(CKPT_DIR, "aa_best.npy"))   # poids de la course
    rng = np.random.default_rng(seed)

    leaf_value = None                  # itération 0 : expert = valeur d'origine
    w2 = None
    for it in range(1, iterations + 1):
        expert = SearchPolicy(w_lin, depth=depth, n_worlds=4, leaf_value=leaf_value)
        print(f"[iteration {it}] generation de {games} parties par l'expert...")
        X, y = build_dataset(expert, games, seed0=int(rng.integers(1_000_000)))
        w2, acc = fit_logistic(X, y)
        print(f"  {len(y)} exemples | precision de prediction du gagnant : {acc*100:.1f}%")
        new_leaf = make_leaf_value(w2)
        wr = evaluate(new_leaf, leaf_value, w_lin, depth=depth, n=40)
        print(f"  valeur v2 (iter {it}) vs valeur precedente, en recherche d{depth} : "
              f"{wr*100:.1f}%")
        leaf_value = new_leaf
        np.save(os.path.join(CKPT_DIR, "aa_value2.npy"), w2)

    print("Valeur enrichie apprise :", np.round(w2, 2))
    return w2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Expert-iteration de la fonction de valeur.")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--depth", type=int, default=2)
    args = p.parse_args()
    train(iterations=args.iterations, games=args.games, depth=args.depth)
