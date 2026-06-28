"""Déroulé d'un match entre deux politiques (jeu glouton, sans exploration).

Réutilisé par `render.py`/`tournament.py` (pour fabriquer un GIF) et par les
stats. La raquette de gauche et celle de droite peuvent être pilotées par deux
tables Q différentes -> c'est ça, « faire s'affronter » deux versions.
"""

from env import (PongEnv, HIT_LEFT, HIT_RIGHT, MISS_LEFT, MISS_RIGHT)
from agent import encode
from render import render_frame


def run_match(left_agent, right_agent, to_points=7, max_steps=20000,
              collect_frames=False, labels=("", ""), seed=0, frame_skip=2,
              max_frames=500):
    """Joue un match jusqu'à `to_points`. Renvoie (score, stats, frames).

    `frame_skip` : on ne filme qu'un pas sur N pour garder le GIF léger.
    """
    env = PongEnv(seed=seed)
    score = [0, 0]                 # [gauche, droite]
    rallies, rally_len = [], 0
    frames = []

    serve_right = True
    env.reset(serve_to_right=serve_right)

    for step in range(max_steps):
        sL = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
                    env.left_y, "left")
        sR = encode(env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
                    env.right_y, "right")
        aL = left_agent.act(sL, greedy=True)
        aR = right_agent.act(sR, greedy=True)

        event = env.step(aL, aR)

        if event in (HIT_LEFT, HIT_RIGHT):
            rally_len += 1

        if collect_frames and (step % frame_skip == 0 or event in (MISS_LEFT, MISS_RIGHT)):
            frames.append(render_frame(
                env, left_label=labels[0], right_label=labels[1],
                score=score, sub=f"echange : {rally_len} renvois"))

        if event in (MISS_LEFT, MISS_RIGHT):
            if event == MISS_LEFT:
                score[1] += 1            # la droite marque
            else:
                score[0] += 1            # la gauche marque
            rallies.append(rally_len)
            rally_len = 0
            if max(score) >= to_points:
                break
            serve_right = not serve_right
            env.reset(serve_to_right=serve_right)

        # garde-fou : deux versions fortes peuvent echanger sans fin
        if collect_frames and len(frames) >= max_frames:
            break

    stats = {
        "points": tuple(score),
        "rallies": len(rallies),
        "avg_rally": sum(rallies) / max(1, len(rallies)),
        "max_rally": max(rallies) if rallies else 0,
    }
    return tuple(score), stats, frames
