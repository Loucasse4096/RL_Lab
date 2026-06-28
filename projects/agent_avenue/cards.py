"""Définition des cartes Agent du jeu Agent Avenue (mode simple).

Chaque type de carte a 3 « paliers » de déplacement selon le nombre
d'exemplaires de ce nom qu'on possède en jeu (1 / 2 / 3 et plus). Au-delà de 3,
on traite comme le 3e palier.

Deux cartes ont un effet spécial sur leur 3e palier :
  - Codebreaker : avoir 3 Codebreakers en jeu fait GAGNER la partie.
  - Daredevil   : avoir 3 Daredevils en jeu fait PERDRE la partie.
"""

# name -> (déplacements [1 copie, 2 copies, 3+ copies], nb d'exemplaires)
CARD_DEFS = {
    "Double Agent": ([-1, 6, -1], 6),
    "Enforcer":     ([1, 2, 3], 6),
    "Sentinel":     ([0, 2, 6], 6),
    "Saboteur":     ([-1, -1, -2], 6),
    "Codebreaker":  ([0, 0, 0], 6),   # 3 en jeu => victoire
    "Daredevil":    ([2, 3, 0], 6),   # 3 en jeu => défaite (3e icone = X)
    "Sidekick":     ([4], 1),         # unique
    "Mole":         ([-3], 1),        # unique
}

CARD_NAMES = list(CARD_DEFS.keys())
NAME_TO_ID = {n: i for i, n in enumerate(CARD_NAMES)}
N_TYPES = len(CARD_NAMES)

WIN_AT_3 = "Codebreaker"
LOSE_AT_3 = "Daredevil"


def move_for(name, count):
    """Déplacement obtenu en recrutant `name` quand on en possède `count` (>=1)."""
    table, _ = CARD_DEFS[name]
    tier = min(count, 3) - 1
    tier = min(tier, len(table) - 1)     # Sidekick/Mole n'ont qu'un palier
    return table[tier]


def build_deck():
    """Renvoie la liste des 38 cartes (noms) composant le paquet Agent."""
    deck = []
    for name, (_, qty) in CARD_DEFS.items():
        deck.extend([name] * qty)
    return deck
