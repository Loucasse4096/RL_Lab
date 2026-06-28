"""Rendu d'une partie d'Agent Avenue en images (Pillow, sans écran).

On dessine :
  - le plateau en boucle avec les deux pions (qui se poursuivent dans le sens
    horaire) ;
  - un panneau par joueur : distance pour rattraper, progression Codebreaker
    (3 = victoire) et Daredevil (3 = défaite), agents recrutés ;
  - une bannière avec le dernier coup joué.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env import BOARD_N, START_GAP
from cards import CARD_NAMES

W_PX, H_PX = 620, 460
BG = (12, 14, 22)
FG = (235, 238, 245)
MUTED = (140, 146, 164)
P_COL = [(90, 200, 255), (255, 140, 120)]     # joueur 0 / joueur 1
GOLD = (255, 205, 90)
DANGER = (240, 90, 90)
RING = (52, 58, 78)


def _font(size, bold=True):
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    path = base + ("-Bold.ttf" if bold else ".ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


F_BIG = _font(22)
F = _font(15)
F_SM = _font(12, bold=False)
F_TINY = _font(11, bold=False)


def _ring_point(cx, cy, r, idx):
    ang = math.radians(-90 + 360.0 * idx / BOARD_N)   # 0 en haut, sens horaire
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


def _draw_board(d, env, cx, cy, r):
    d.ellipse([cx - r - 16, cy - r - 16, cx + r + 16, cy + r + 16],
              outline=RING, width=2)
    for i in range(BOARD_N):
        x, y = _ring_point(cx, cy, r, i)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=RING)
    # cases maison
    for p, home in enumerate((0, START_GAP)):
        x, y = _ring_point(cx, cy, r, home)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=P_COL[p], width=2)
    # pions
    for p in (0, 1):
        x, y = _ring_point(cx, cy, r, env.phys_pos(p))
        off = -10 if p == 0 else 10
        d.ellipse([x - 11, y - 11 + off * 0, x + 11, y + 11], fill=P_COL[p],
                  outline=(15, 17, 26), width=2)
        d.text((x - 4, y - 8), str(p), font=F, fill=(15, 17, 26))
    d.text((cx - 22, cy - 8), "loop", font=F_TINY, fill=MUTED)


def _pips(d, x, y, filled, total=3, col=GOLD):
    for i in range(total):
        c = col if i < filled else RING
        d.ellipse([x + i * 16, y, x + i * 16 + 11, y + 11], fill=c)


def _panel(d, env, p, x, y):
    col = P_COL[p]
    d.text((x, y), f"joueur {p}", font=F, fill=col)
    # distance pour rattraper (barre : pleine = sur le point de rattraper)
    dist = env.catch_distance(p)
    prog = max(0.0, min(1.0, (BOARD_N - dist) / BOARD_N))
    bx, by, bw = x, y + 22, 150
    d.rounded_rectangle([bx, by, bx + bw, by + 10], radius=4, fill=RING)
    d.rounded_rectangle([bx, by, bx + int(bw * prog), by + 10], radius=4, fill=col)
    d.text((bx + bw + 8, by - 2), f"-{dist}", font=F_SM, fill=MUTED)

    counts = env.counts[p]
    cb = int(counts[CARD_NAMES.index("Codebreaker")])
    dd = int(counts[CARD_NAMES.index("Daredevil")])
    d.text((x, y + 40), "Codebreaker", font=F_TINY, fill=MUTED)
    _pips(d, x + 84, y + 40, min(cb, 3), col=GOLD)
    d.text((x, y + 56), "Daredevil", font=F_TINY, fill=MUTED)
    _pips(d, x + 84, y + 56, min(dd, 3), col=DANGER)

    # agents recrutés
    parts = []
    for i, name in enumerate(CARD_NAMES):
        n = int(counts[i])
        if n:
            parts.append(f"{name[:4]}x{n}")
    txt = "  ".join(parts) if parts else "(aucun agent)"
    d.text((x, y + 76), txt, font=F_TINY, fill=(180, 186, 200))


def render_frame(env, info, labels=("", "")):
    img = Image.new("RGB", (W_PX, H_PX), BG)
    d = ImageDraw.Draw(img)

    # bannière
    title = "Agent Avenue"
    if labels[0] or labels[1]:
        title = f"{labels[0]}   vs   {labels[1]}"
    d.text((20, 14), title, font=F_BIG, fill=FG)
    d.text((20, 44), f"tour {env.turn}", font=F_SM, fill=MUTED)

    _draw_board(d, env, cx=170, cy=250, r=120)
    _panel(d, env, 0, x=360, y=80)
    _panel(d, env, 1, x=360, y=250)

    # dernier coup
    if info is not None:
        a, o = info["active"], info["opp"]
        line = (f"J{a} recrute {info['act_card']} ({info['m_act']:+d})   |   "
                f"J{o} recrute {info['opp_card']} ({info['m_opp']:+d})")
        d.text((20, H_PX - 30), line, font=F_SM, fill=FG)

    if env.winner is not None:
        wcol = P_COL[env.winner]
        d.rectangle([0, H_PX // 2 - 26, W_PX, H_PX // 2 + 26], fill=(0, 0, 0))
        msg = f"  joueur {env.winner} ({labels[env.winner] or '?'}) GAGNE !"
        d.text((W_PX // 2 - 150, H_PX // 2 - 14), msg, font=F_BIG, fill=wcol)

    return img


def save_gif(frames, path, fps=2, hold_last=6):
    if not frames:
        raise ValueError("aucune image")
    frames = list(frames) + [frames[-1]] * hold_last      # on tient la fin
    duration = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    return path
