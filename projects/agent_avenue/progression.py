"""Visualisations de la progression de l'apprentissage (PNG via Pillow).

1. learning_curve.png : taux de victoire du champion au fil des générations,
   contre la version naïve (gen 0) et contre l'aléatoire. C'est la courbe
   « regarde-le apprendre ».
2. ladder.png : matrice des duels entre les checkpoints sauvegardés
   (qui bat qui, et de combien) — pour « les faire s'affronter ».

Usage :
    python progression.py
"""

import glob
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from policies import ValuePolicy
from match import run_game

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")
BG = (12, 14, 22)
FG = (235, 238, 245)
MUTED = (140, 146, 164)
GRID = (40, 45, 62)
GOLD = (255, 205, 90)
BLUE = (90, 200, 255)


def _font(size, bold=True):
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    try:
        return ImageFont.truetype(base + ("-Bold.ttf" if bold else ".ttf"), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F, F_SM = _font(20), _font(14), _font(12, bold=False)


def learning_curve(out):
    hist = np.load(os.path.join(CKPT_DIR, "history.npy"))
    gens = hist[:, 0]
    vs_gen0, vs_rand = hist[:, 1] * 100, hist[:, 2] * 100
    gmax = max(1, gens.max())

    W, H = 660, 430
    L, R, T, B = 70, 30, 70, 60          # marges
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((L - 10, 22), "Agent Avenue — progression de l'apprentissage",
           font=F_TITLE, fill=FG)

    def px(g):
        return L + (W - L - R) * g / gmax

    def py(p):
        return T + (H - T - B) * (1 - p / 100.0)

    for p in (0, 25, 50, 75, 100):
        y = py(p)
        d.line([(L, y), (W - R, y)], fill=GRID, width=1)
        d.text((L - 40, y - 7), f"{p}%", font=F_SM, fill=MUTED)
    d.line([(L, py(50)), (W - R, py(50))], fill=(80, 86, 104), width=1)  # repere 50%

    for g in range(0, int(gmax) + 1, max(1, int(gmax) // 8)):
        d.text((px(g) - 6, H - B + 8), str(g), font=F_SM, fill=MUTED)
    d.text(((W) / 2 - 40, H - 24), "generation", font=F_SM, fill=MUTED)

    def plot(ys, col):
        pts = [(px(g), py(v)) for g, v in zip(gens, ys)]
        d.line(pts, fill=col, width=3, joint="curve")
        for x, y in pts:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)

    plot(vs_gen0, GOLD)
    plot(vs_rand, BLUE)

    # legende
    d.rectangle([L, 50, L + 14, 58], fill=GOLD)
    d.text((L + 20, 47), "vs version naive (gen 0)", font=F_SM, fill=FG)
    d.rectangle([L + 220, 50, L + 234, 58], fill=BLUE)
    d.text((L + 240, 47), "vs jeu aleatoire", font=F_SM, fill=FG)

    img.save(out)
    return out


def _available_gens():
    gens = []
    for p in glob.glob(os.path.join(CKPT_DIR, "aa_gen*.npy")):
        m = re.search(r"aa_gen(\d+)\.npy", os.path.basename(p))
        if m:
            gens.append(int(m.group(1)))
    return sorted(gens)


def ladder(out, n_games=120):
    gens = _available_gens()
    if len(gens) < 2:
        print("  (pas assez de checkpoints pour la matrice)")
        return None
    pols = {g: ValuePolicy(np.load(os.path.join(CKPT_DIR, f"aa_gen{g}.npy"))) for g in gens}

    n = len(gens)
    M = np.full((n, n), 0.5)
    for i, gi in enumerate(gens):
        for j, gj in enumerate(gens):
            if i == j:
                continue
            wa = 0
            for k in range(n_games):
                w, _, _ = run_game(pols[gi], pols[gj], seed=k)
                wa += (w == 0)
            M[i, j] = wa / n_games

    cell, pad, top, left = 54, 8, 90, 90
    W = left + n * cell + pad
    H = top + n * cell + pad + 30
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((20, 24), "Duels entre versions — % de victoire (ligne vs colonne)",
           font=F, fill=FG)

    for k, g in enumerate(gens):
        d.text((left + k * cell + cell // 2 - 12, top - 22), f"g{g}", font=F_SM, fill=BLUE)
        d.text((left - 40, top + k * cell + cell // 2 - 7), f"g{g}", font=F_SM, fill=GOLD)

    for i in range(n):
        for j in range(n):
            x0, y0 = left + j * cell, top + i * cell
            if i == j:
                col = (35, 39, 54)
            else:
                v = M[i, j]
                # rouge (perd) -> vert (gagne)
                r = int(220 * (1 - v) + 40 * v)
                gr = int(60 * (1 - v) + 200 * v)
                col = (r, gr, 70)
            d.rounded_rectangle([x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2],
                                radius=6, fill=col)
            if i != j:
                d.text((x0 + cell // 2 - 12, y0 + cell // 2 - 7),
                       f"{int(round(M[i, j]*100))}", font=F_SM, fill=(15, 17, 26))
    img.save(out)
    return out


def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    c = learning_curve(os.path.join(MEDIA_DIR, "learning_curve.png"))
    print(f"  Courbe d'apprentissage : {c}")
    l = ladder(os.path.join(MEDIA_DIR, "ladder.png"))
    if l:
        print(f"  Matrice des duels : {l}")


if __name__ == "__main__":
    main()
