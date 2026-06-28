"""Démonstration de la montée en intelligence de l'agent.

On compare plusieurs niveaux d'agent, tous mesurés au même étalon : l'agent
évolué d'origine (valeur linéaire, 1 coup d'anticipation).

  - évolué (1 coup)                  : la référence
  - recherche d2 / d3                : anticipation plus profonde (minimax + PIMC)
  - recherche + valeur apprise       : en plus, une évaluation plus fine apprise
                                       par expert-iteration

Produit :
  - media/intelligence.png : taux de victoire de chaque niveau contre la référence
  - media/duel_smart_vs_evolved.gif : l'agent le plus malin contre l'évolué
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from policies import ValuePolicy
from search import SearchPolicy
from smart_value import make_leaf_value
from match import run_game
from render import save_gif

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")
BG = (12, 14, 22)
FG = (235, 238, 245)
MUTED = (140, 146, 164)
GRID = (40, 45, 62)
BAR = (90, 200, 255)
BAR_HI = (120, 230, 150)


def _font(size, bold=True):
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    try:
        return ImageFont.truetype(base + ("-Bold.ttf" if bold else ".ttf"), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F, F_SM = _font(19), _font(14), _font(12, bold=False)


def winrate(contender, baseline, n=40):
    wa = 0
    for i in range(n):
        r, _, _ = run_game(contender, baseline, seed=i); wa += (r == 0)
        r, _, _ = run_game(baseline, contender, seed=6000 + i); wa += (r == 1)
    return wa / (2 * n)


def bar_chart(rows, out):
    """rows = [(label, winrate, highlight_bool)]."""
    W, H = 760, 360
    L, R, T, B = 300, 40, 70, 50
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 22), "Agent Avenue — montee en intelligence", font=F_TITLE, fill=FG)
    d.text((24, 46), "% de victoire contre l'agent evolue d'origine (1 coup)",
           font=F_SM, fill=MUTED)

    plot_w = W - L - R
    for pct in (0, 25, 50, 75, 100):
        x = L + plot_w * pct / 100
        d.line([(x, T), (x, H - B)], fill=GRID, width=1)
        d.text((x - 8, H - B + 8), f"{pct}%", font=F_SM, fill=MUTED)
    x50 = L + plot_w * 0.5
    d.line([(x50, T), (x50, H - B)], fill=(120, 126, 150), width=2)
    d.text((x50 - 96, T - 16), "evolue d'origine (50%)", font=F_SM, fill=(150, 156, 174))

    n = len(rows)
    band = (H - T - B) / n
    for i, (label, wr, hi) in enumerate(rows):
        y = T + i * band + band * 0.2
        h = band * 0.6
        col = BAR_HI if hi else BAR
        d.text((24, y + h / 2 - 8), label, font=F, fill=FG)
        d.rounded_rectangle([L, y, L + plot_w * wr, y + h], radius=5, fill=col)
        d.text((L + plot_w * wr + 8, y + h / 2 - 8), f"{wr*100:.0f}%", font=F, fill=col)
    img.save(out)
    return out


def main():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    w_lin = np.load(os.path.join(CKPT_DIR, "aa_best.npy"))
    leaf2 = make_leaf_value(np.load(os.path.join(CKPT_DIR, "aa_value2.npy")))

    baseline = ValuePolicy(w_lin)
    contenders = [
        ("recherche profondeur 2", SearchPolicy(w_lin, depth=2), False),
        ("recherche d2 + valeur apprise", SearchPolicy(w_lin, depth=2, leaf_value=leaf2), False),
        ("recherche d3 + valeur apprise", SearchPolicy(w_lin, depth=3, leaf_value=leaf2), True),
    ]
    rows = []
    for label, pol, hi in contenders:
        wr = winrate(pol, baseline, n=60)
        print(f"  {label:36s} : {wr*100:5.1f}%")
        rows.append((label, wr, hi))
    out = bar_chart(rows, os.path.join(MEDIA_DIR, "intelligence.png"))
    print(f"  Graphe : {out}")

    # duel filme : le plus malin vs l'evolue
    smart = SearchPolicy(w_lin, depth=3, n_worlds=6, leaf_value=leaf2)
    w, st, frames = run_game(smart, baseline, seed=4, render=True,
                             labels=("malin (d3+appris)", "evolue"))
    gif = save_gif(frames, os.path.join(MEDIA_DIR, "duel_smart_vs_evolved.gif"), fps=2)
    print(f"  Duel filme : gagnant J{w} en {st['turns']} tours -> {gif}")


if __name__ == "__main__":
    main()
