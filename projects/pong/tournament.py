"""Fait s'affronter deux versions entrainees et rend le match en GIF.

Usage :
    python tournament.py 1000 20000           # v1000 (gauche) vs v20000 (droite)
    python tournament.py 1000 20000 --points 7 --out media/duel.gif
"""

import argparse
import os

from agent import QAgent
from match import run_match
from render import save_gif

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")


def load(n):
    path = os.path.join(CKPT_DIR, f"pong_{n}.npy")
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint introuvable : {path}\n"
                         f"Lance d'abord :  python train.py")
    return QAgent.load(path)


def main():
    p = argparse.ArgumentParser(description="Duel entre deux versions du Pong.")
    p.add_argument("left", type=int, help="iteration de la version de gauche")
    p.add_argument("right", type=int, help="iteration de la version de droite")
    p.add_argument("--points", type=int, default=5)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-gif", action="store_true", help="stats seulement")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    left, right = load(args.left), load(args.right)
    labels = (f"v{args.left}", f"v{args.right}")

    score, stats, frames = run_match(
        left, right, to_points=args.points, frame_skip=3,
        collect_frames=not args.no_gif, labels=labels)

    print(f"\n  {labels[0]}  {score[0]} - {score[1]}  {labels[1]}")
    gagnant = labels[0] if score[0] > score[1] else labels[1]
    print(f"  Vainqueur : {gagnant}")
    print(f"  Echanges joues : {stats['rallies']}  |  "
          f"echange moyen : {stats['avg_rally']:.1f}  |  "
          f"plus long : {stats['max_rally']}")

    if not args.no_gif:
        os.makedirs(MEDIA_DIR, exist_ok=True)
        out = args.out or os.path.join(MEDIA_DIR, f"duel_{args.left}_vs_{args.right}.gif")
        save_gif(frames, out, fps=args.fps)
        print(f"  GIF : {out}  ({len(frames)} images)")


if __name__ == "__main__":
    main()
