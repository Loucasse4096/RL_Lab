"""Entraînement en self-play du Pong par Q-learning tabulaire.

Une seule table Q apprend, en jouant des deux côtés du terrain (la gauche voit
le monde en miroir). On sauvegarde des « checkpoints » à différents nombres de
manches jouées : ce sont eux qu'on fera ensuite s'affronter / qu'on regardera
progresser.

Usage :
    python train.py                          # schéma de checkpoints par défaut
    python train.py --checkpoints 0 500 5000 50000
"""

import argparse
import os

import numpy as np

from env import PongEnv, HIT_LEFT, HIT_RIGHT, MISS_LEFT, MISS_RIGHT
from agent import QAgent, encode

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def reward(event, side):
    """Récompense dense : +1 quand on renvoie, -1 quand on encaisse."""
    if side == "left":
        if event == HIT_LEFT:
            return 1.0, False
        if event == MISS_LEFT:
            return -1.0, True
    else:
        if event == HIT_RIGHT:
            return 1.0, False
        if event == MISS_RIGHT:
            return -1.0, True
    # une fin de manche est terminale pour les deux raquettes
    if event in (MISS_LEFT, MISS_RIGHT):
        return 0.0, True
    return 0.0, False


def train(checkpoints, eps_start=0.25, eps_end=0.02, seed=0, log_every=2000):
    total = max(checkpoints)
    agent = QAgent(epsilon=eps_start, seed=seed)
    env = PongEnv(seed=seed)
    env.reset()

    os.makedirs(CKPT_DIR, exist_ok=True)
    saved = set()
    recent_rallies = []
    rally_len = 0
    rallies_done = 0

    def maybe_save(n):
        if n in checkpoints and n not in saved:
            path = os.path.join(CKPT_DIR, f"pong_{n}.npy")
            agent.save(path)
            saved.add(n)
            avg = np.mean(recent_rallies[-500:]) if recent_rallies else 0.0
            print(f"  checkpoint {n:>6} manches  |  echange moyen recent : {avg:4.1f}")

    maybe_save(0)                      # politique « zero iteration » (aléatoire)

    while rallies_done < total:
        # fraction de l'entrainement ecoulee -> decroissance d'epsilon
        frac = rallies_done / total
        agent.epsilon = eps_start + (eps_end - eps_start) * frac

        sL = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
                    env.left_y, "left")
        sR = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
                    env.right_y, "right")
        aL = agent.act(sL)
        aR = agent.act(sR)

        event = env.step(aL, aR)

        rL, tL = reward(event, "left")
        rR, tR = reward(event, "right")
        terminal = tL or tR

        sL2 = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
                     env.left_y, "left")
        sR2 = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
                     env.right_y, "right")
        agent.update(sL, aL, rL, sL2, terminal)
        agent.update(sR, aR, rR, sR2, terminal)

        if event in (HIT_LEFT, HIT_RIGHT):
            rally_len += 1

        if terminal:
            recent_rallies.append(rally_len)
            rally_len = 0
            rallies_done += 1
            maybe_save(rallies_done)
            if rallies_done % log_every == 0:
                avg = np.mean(recent_rallies[-500:])
                print(f"  ... {rallies_done:>6}/{total}  echange moyen : {avg:4.1f}")
            env.reset()

    # garantit que tous les checkpoints demandes existent
    for n in checkpoints:
        maybe_save(n)
    print("Entrainement termine.")
    return agent


def parse_args():
    p = argparse.ArgumentParser(description="Entraine le Pong en self-play.")
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[0, 200, 1000, 5000, 20000],
                   help="nombres de manches auxquels sauvegarder une version")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Checkpoints vises : {sorted(set(args.checkpoints))}")
    train(sorted(set(args.checkpoints)), seed=args.seed)
