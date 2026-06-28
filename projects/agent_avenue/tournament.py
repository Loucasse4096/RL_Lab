"""Fait s'affronter deux versions entraînées et rend une partie en GIF.

Usage :
    python tournament.py 0 25            # gen0 vs gen25
    python tournament.py 5 40 --games 200 --out media/duel.gif
"""

import argparse
import os

import numpy as np

from policies import ValuePolicy
from match import run_game
from render import save_gif

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")


def load_gen(n):
    path = os.path.join(CKPT_DIR, f"aa_gen{n}.npy")
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint introuvable : {path}\nLance d'abord : python train.py")
    return np.load(path)


def winrate(a, b, n_games):
    """Taux de victoire de A contre B, joué des deux côtés (sans biais)."""
    pa, pb = ValuePolicy(a), ValuePolicy(b)
    wa = 0
    for i in range(n_games):
        w, _, _ = run_game(pa, pb, seed=i)
        wa += (w == 0)
        w, _, _ = run_game(pb, pa, seed=10000 + i)
        wa += (w == 1)
    return wa / (2 * n_games)


def main():
    p = argparse.ArgumentParser(description="Duel entre deux versions d'Agent Avenue.")
    p.add_argument("a", type=int, help="generation de la version A")
    p.add_argument("b", type=int, help="generation de la version B")
    p.add_argument("--games", type=int, default=200, help="parties par cote pour le score")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--seed", type=int, default=7, help="graine de la partie filmee")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    a, b = load_gen(args.a), load_gen(args.b)
    la, lb = f"gen{args.a}", f"gen{args.b}"

    wr = winrate(a, b, args.games)
    print(f"\n  {la}  vs  {lb}   ({2*args.games} parties)")
    print(f"  {la} gagne {wr*100:.1f}%   |   {lb} gagne {(1-wr)*100:.1f}%")
    print(f"  Vainqueur du duel : {la if wr >= 0.5 else lb}")

    if not args.no_gif:
        w, st, frames = run_game(ValuePolicy(a), ValuePolicy(b),
                                 seed=args.seed, render=True, labels=(la, lb))
        os.makedirs(MEDIA_DIR, exist_ok=True)
        out = args.out or os.path.join(MEDIA_DIR, f"duel_gen{args.a}_vs_gen{args.b}.gif")
        save_gif(frames, out, fps=2)
        print(f"  Partie filmee : {la if w == 0 else lb} gagne en {st['turns']} tours "
              f"({st['reason']}) -> {out}")


if __name__ == "__main__":
    main()
