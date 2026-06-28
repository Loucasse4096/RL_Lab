"""Rendu d'une partie en GIF avec Pillow (aucun écran requis).

On dessine chaque image de jeu dans un tableau RGB agrandi, puis on assemble le
tout en GIF animé. Une petite bannière en haut affiche les labels (numéro
d'itération, score...).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env import (WIDTH, HEIGHT, PADDLE_HALF, PADDLE_X_LEFT, PADDLE_X_RIGHT,
                 BALL_RADIUS)

CELL = 12                       # pixels par unité de grille
BANNER = 34                     # hauteur de la bannière (px)
W_PX = int(WIDTH * CELL)
H_PX = int(HEIGHT * CELL)

BG = (12, 14, 22)
FG = (235, 238, 245)
ACCENT = (90, 200, 255)
BALL_COL = (255, 210, 80)
NET = (60, 66, 84)


def _font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = _font(18)
FONT_SM = _font(13)


def render_frame(env, left_label="", right_label="", score=None, sub=""):
    """Construit une image PIL de l'état courant de l'environnement."""
    img = Image.new("RGB", (W_PX, H_PX + BANNER), BG)
    d = ImageDraw.Draw(img)

    # Bannière
    if left_label:
        d.text((12, 7), left_label, font=FONT, fill=ACCENT)
    if right_label:
        w = d.textlength(right_label, font=FONT)
        d.text((W_PX - 12 - w, 7), right_label, font=FONT, fill=(255, 140, 120))
    if score is not None:
        s = f"{score[0]} : {score[1]}"
        w = d.textlength(s, font=FONT)
        d.text(((W_PX - w) / 2, 7), s, font=FONT, fill=FG)
    if sub:
        w = d.textlength(sub, font=FONT_SM)
        d.text(((W_PX - w) / 2, BANNER - 14), sub, font=FONT_SM, fill=(150, 156, 174))

    oy = BANNER

    # Filet central
    for y in range(0, H_PX, 24):
        d.line([(W_PX // 2, oy + y), (W_PX // 2, oy + y + 12)], fill=NET, width=2)

    # Raquettes
    def paddle(cx, cy, col):
        x = cx * CELL
        y0 = oy + (cy - PADDLE_HALF) * CELL
        y1 = oy + (cy + PADDLE_HALF) * CELL
        d.rounded_rectangle([x - 5, y0, x + 5, y1], radius=4, fill=col)

    paddle(PADDLE_X_LEFT, env.left_y, ACCENT)
    paddle(PADDLE_X_RIGHT, env.right_y, (255, 140, 120))

    # Balle
    bx = env.ball_x * CELL
    by = oy + env.ball_y * CELL
    r = BALL_RADIUS * CELL + 2
    d.ellipse([bx - r, by - r, bx + r, by + r], fill=BALL_COL)

    return img


def save_gif(frames, path, fps=30):
    """Enregistre une liste d'images PIL en GIF animé."""
    if not frames:
        raise ValueError("aucune image à enregistrer")
    duration = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    return path
