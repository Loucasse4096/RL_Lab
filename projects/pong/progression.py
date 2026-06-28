"""GIF « regarde-le progresser » facon video YouTube.

Pour chaque checkpoint (0, 200, 1000, ...), on laisse la version jouer contre
elle-meme pendant quelques echanges et on filme. On enchaine tout dans un seul
GIF avec le compteur d'iterations affiche en gros. On voit la balle passer de
« ratee tout de suite » a « echanges interminables ».

Usage :
    python progression.py
    python progression.py --versions 0 1000 20000 --rallies 3
"""

import argparse
import glob
import os
import re

from agent import QAgent
from env import (PongEnv, HIT_LEFT, HIT_RIGHT, MISS_LEFT, MISS_RIGHT)
from agent import encode
from render import render_frame, save_gif

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")


def available_versions():
    out = []
    for p in glob.glob(os.path.join(CKPT_DIR, "pong_*.npy")):
        m = re.search(r"pong_(\d+)\.npy", os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def play_clip(agent, n_rallies, version_label, seed=0, frame_skip=2, max_frames=140):
    """Filme l'agent jouant contre lui-meme.

    S'arrete des que `n_rallies` echanges sont joues OU que `max_frames` images
    sont filmees (pour qu'une version tres forte ne fasse pas exploser le GIF).
    """
    env = PongEnv(seed=seed)
    env.reset(serve_to_right=True)
    frames = []
    done = 0
    rally_len = 0
    max_steps = 8000
    for step in range(max_steps):
        sL = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy, env.left_y, "left")
        sR = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy, env.right_y, "right")
        event = env.step(agent.act(sL, greedy=True), agent.act(sR, greedy=True))
        if event in (HIT_LEFT, HIT_RIGHT):
            rally_len += 1
        if step % frame_skip == 0 or event in (MISS_LEFT, MISS_RIGHT):
            frames.append(render_frame(
                env, left_label=version_label, right_label="self-play",
                sub=f"echange en cours : {rally_len} renvois"))
        if event in (MISS_LEFT, MISS_RIGHT):
            done += 1
            rally_len = 0
            if done >= n_rallies:
                break
            env.reset(serve_to_right=(done % 2 == 0))
        if len(frames) >= max_frames:
            break
    return frames


def main():
    p = argparse.ArgumentParser(description="GIF de progression de l'apprentissage.")
    p.add_argument("--versions", type=int, nargs="+", default=None,
                   help="iterations a enchainer (defaut : tous les checkpoints)")
    p.add_argument("--rallies", type=int, default=3,
                   help="echanges filmes par version")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=os.path.join(MEDIA_DIR, "progression.gif"))
    args = p.parse_args()

    versions = args.versions or available_versions()
    if not versions:
        raise SystemExit("Aucun checkpoint. Lance d'abord :  python train.py")

    all_frames = []
    for n in versions:
        path = os.path.join(CKPT_DIR, f"pong_{n}.npy")
        if not os.path.exists(path):
            print(f"  (ignore v{n} : checkpoint absent)")
            continue
        agent = QAgent.load(path)
        label = f"iteration {n}"
        print(f"  filme {label} ...")
        all_frames.extend(play_clip(agent, args.rallies, label))

    os.makedirs(MEDIA_DIR, exist_ok=True)
    save_gif(all_frames, args.out, fps=args.fps)
    print(f"\n  GIF de progression : {args.out}  ({len(all_frames)} images)")


if __name__ == "__main__":
    main()
