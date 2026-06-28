"""Joue toi-même une partie d'Agent Avenue (mode simple) contre un agent.

Au terminal : tu vois le plateau, ta main, et tu choisis tes coups. L'agent
joue de l'autre côté. À la fin, la partie est enregistrée en GIF dans media/.

Usage :
    python play.py                    # contre l'agent le plus malin (recherche + valeur apprise)
    python play.py --opponent gen40   # contre une version evoluee precise
    python play.py --opponent gen0    # contre la version naive
    python play.py --first            # tu commences
"""

import argparse
import os
import random

import numpy as np

from env import AgentAvenue, BOARD_N
from cards import CARD_NAMES, NAME_TO_ID
from policies import ValuePolicy
from search import SearchPolicy
from smart_value import make_leaf_value
from render import render_frame, save_gif

NAME_CB = NAME_TO_ID["Codebreaker"]
NAME_DD = NAME_TO_ID["Daredevil"]

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")


# --------------------------------------------------------------------------
# Chargement de l'adversaire
# --------------------------------------------------------------------------
def load_opponent(name, depth):
    if name == "smart":
        w = np.load(os.path.join(CKPT_DIR, "aa_best.npy"))
        leaf = make_leaf_value(np.load(os.path.join(CKPT_DIR, "aa_value2.npy")))
        return SearchPolicy(w, depth=depth, n_worlds=6, leaf_value=leaf), \
            f"agent malin (recherche d{depth} + valeur apprise)"
    if name.startswith("gen"):
        path = os.path.join(CKPT_DIR, f"aa_{name}.npy")
        if not os.path.exists(path):
            raise SystemExit(f"Checkpoint introuvable : {path}")
        return ValuePolicy(np.load(path)), f"agent evolue {name}"
    raise SystemExit(f"Adversaire inconnu : {name} (essaie smart, gen0, gen40...)")


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
def names_label(p):
    return "TOI" if p == HUMAN else "AGENT"


def show_state(env, last=""):
    print("\n" + "=" * 56)
    if last:
        print(last)
    print(f"  Tour {env.turn}   (boucle de {BOARD_N} cases)")
    for p in (0, 1):
        d = env.catch_distance(p)
        cb = int(env.counts[p][NAME_CB])
        dd = int(env.counts[p][NAME_DD])
        agents = "  ".join(f"{n[:4]}x{int(env.counts[p][i])}"
                           for i, n in enumerate(CARD_NAMES) if env.counts[p][i])
        tag = names_label(p)
        print(f"  [{tag:5}] doit avancer de {d:2d} pour rattraper  | "
              f"Codebreaker {min(cb,3)}/3  Daredevil {min(dd,3)}/3")
        if agents:
            print(f"          agents : {agents}")
    print("=" * 56)


def ask_int(prompt, valid):
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) in valid:
            return int(raw)
        print(f"  -> entre un nombre parmi {sorted(valid)}")


def human_discards(env):
    while env.discards_left[HUMAN] > 0 and env.deck:
        hand = env.hands[HUMAN]
        print(f"\n  Ta main : " + "  ".join(f"[{i}] {c}" for i, c in enumerate(hand)))
        print(f"  Defausses restantes : {env.discards_left[HUMAN]}")
        raw = input("  Defausser une carte ? (numero, ou Entree pour continuer) : ").strip()
        if raw == "":
            return
        if raw.isdigit() and 0 <= int(raw) < len(hand):
            removed = hand[int(raw)]
            env.discard(HUMAN, removed)
            print(f"  -> defausse {removed}, pioche une nouvelle carte.")
        else:
            print("  -> entree invalide")


def human_play(env):
    human_discards(env)
    legal = set(env.legal_plays(HUMAN))
    while True:
        hand = env.hands[HUMAN]
        print("\n  Ta main : " + "  ".join(f"[{i}] {c}" for i, c in enumerate(hand)))
        print("  Joue 2 cartes (noms differents) : 1 face VISIBLE, 1 face CACHEE.")
        i = ask_int("  Carte face VISIBLE  [index] : ", set(range(len(hand))))
        j = ask_int("  Carte face CACHEE   [index] : ", set(range(len(hand))) - {i})
        up, down = hand[i], hand[j]
        if (up, down) in legal:
            return up, down
        print("  -> les deux cartes doivent avoir des noms differents. Reessaie.")


def human_recruit(env, up):
    print(f"\n  L'agent a joue 2 cartes : une VISIBLE = '{up}', une CACHEE (?).")
    print("  Tu en recrutes UNE (l'agent prend l'autre).")
    raw = input(f"  Prendre la carte visible '{up}' ? (o = visible / n = cachee) : ").strip().lower()
    return raw.startswith("o") or raw.startswith("y") or raw == ""


# --------------------------------------------------------------------------
# Boucle de jeu
# --------------------------------------------------------------------------
def play(opponent, opp_label, human_first, seed):
    env = AgentAvenue(seed=seed)
    bot = opponent
    frames = [render_frame(env, None, LABELS)]
    print(f"\nTu affrontes : {opp_label}")
    print(f"Tu es le joueur {HUMAN}.  But : rattraper le pion adverse sur la boucle,")
    print("ou avoir 3 Codebreakers (victoire) — eviter 3 Daredevils (defaite).")

    winner = None
    last = ""
    for _ in range(200):
        active, opp = env.active, 1 - env.active
        if len(env.hands[active]) < 2:
            d0, d1 = env.catch_distance(0), env.catch_distance(1)
            winner = 0 if d0 < d1 else (1 if d1 < d0 else active)
            env.winner = winner
            break

        if active == HUMAN:
            show_state(env, last)
            up, down = human_play(env)
            recruit_up = bot.choose_recruit(env, up)
            info = env.apply_turn(up, down, recruit_up)
            last = (f"  TOI : tu joues {up} (visible) / {down} (cachee). "
                    f"L'agent recrute {info['opp_card']} ({info['m_opp']:+d}), "
                    f"tu prends {info['act_card']} ({info['m_act']:+d}).")
        else:
            up, down = bot.choose_play(env)
            show_state(env, last)
            recruit_up = human_recruit(env, up)
            info = env.apply_turn(up, down, recruit_up)
            last = (f"  AGENT : tu recrutes {info['opp_card']} ({info['m_opp']:+d}), "
                    f"l'agent prend {info['act_card']} ({info['m_act']:+d}). "
                    f"(la carte cachee etait {down})")

        frames.append(render_frame(env, info, LABELS))
        winner = env.check_end()
        if winner is not None:
            break
        env.active = opp

    show_state(env, last)
    if winner == HUMAN:
        print("\n  🎉 TU GAGNES ! Tu as demasque l'agent.")
    elif winner is None:
        print("\n  Partie nulle / interrompue.")
    else:
        print("\n  😬 L'agent gagne. Reessaie !")

    os.makedirs(MEDIA_DIR, exist_ok=True)
    out = os.path.join(MEDIA_DIR, "ma_partie.gif")
    save_gif(frames, out, fps=2)
    print(f"\n  Rejoue ta partie en image : {out}")


def main():
    global HUMAN, LABELS
    p = argparse.ArgumentParser(description="Joue contre un agent d'Agent Avenue.")
    p.add_argument("--opponent", default="smart",
                   help="smart (defaut), gen0, gen6, gen12, gen20, gen30, gen40")
    p.add_argument("--depth", type=int, default=3, help="profondeur de recherche (agent smart)")
    p.add_argument("--first", action="store_true", help="tu commences (sinon l'agent commence)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    HUMAN = 0 if args.first else 1
    LABELS = ["TOI" if i == HUMAN else "agent" for i in (0, 1)]
    seed = args.seed if args.seed is not None else random.randint(0, 10_000)

    opponent, opp_label = load_opponent(args.opponent, args.depth)
    play(opponent, opp_label, args.first, seed)


if __name__ == "__main__":
    main()
