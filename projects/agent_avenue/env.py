"""Moteur du jeu Agent Avenue — mode simple, 2 joueurs.

Le plateau est une boucle. Plutôt que de suivre deux positions absolues, on suit
l'écart `gap` = distance (sens horaire) du pion 0 vers le pion 1. Le pion 0
rattrape le pion 1 quand gap <= 0 ; le pion 1 rattrape le pion 0 quand gap >= N
(il a bouclé la moitié restante). On garde aussi des positions cumulées, juste
pour l'affichage.

Le moteur ne décide rien : il expose l'état, les coups légaux, et des méthodes
de mutation. Ce sont les politiques (voir policies.py) qui choisissent.

Déroulé d'un tour (joueur actif = celui dont c'est le tour) :
  1. PLAY    : l'actif pose 2 cartes de sa main (1 visible, 1 cachée, noms
               différents). Avant ça, il peut défausser (max 4 fois/partie).
  2. RECRUIT : l'adversaire prend 1 des 2 cartes (il ne voit que la visible),
               l'actif prend l'autre. Chacun avance selon la carte recrutée.
  3. END     : on vérifie victoire / défaite.
"""

import numpy as np

from cards import (CARD_NAMES, NAME_TO_ID, N_TYPES, move_for, build_deck,
                   WIN_AT_3, LOSE_AT_3)

BOARD_N = 16            # taille de la boucle
START_GAP = BOARD_N // 2
HAND_SIZE = 4
MAX_DISCARDS = 4

WIN_CB = NAME_TO_ID[WIN_AT_3]      # 3 Codebreakers => victoire
LOSE_DD = NAME_TO_ID[LOSE_AT_3]    # 3 Daredevils  => défaite


class AgentAvenue:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        deck = build_deck()
        self.rng.shuffle(deck)
        self.deck = list(deck)
        self.hands = [[self.deck.pop() for _ in range(HAND_SIZE)],
                      [self.deck.pop() for _ in range(HAND_SIZE)]]
        self.counts = [np.zeros(N_TYPES, dtype=int), np.zeros(N_TYPES, dtype=int)]
        self.pos_cum = [0, 0]            # progression cumulée (affichage)
        self.discards_left = [MAX_DISCARDS, MAX_DISCARDS]
        self.active = 0
        self.turn = 0
        self.winner = None               # None / 0 / 1 une fois la partie finie
        return self

    def clone(self):
        """Copie profonde légère de l'état (pour la recherche en profondeur)."""
        c = AgentAvenue.__new__(AgentAvenue)
        c.rng = self.rng                 # inutilisé pendant la recherche
        c.deck = list(self.deck)
        c.hands = [list(self.hands[0]), list(self.hands[1])]
        c.counts = [self.counts[0].copy(), self.counts[1].copy()]
        c.pos_cum = list(self.pos_cum)
        c.discards_left = list(self.discards_left)
        c.active = self.active
        c.turn = self.turn
        c.winner = self.winner
        return c

    # ---- lecture d'état ---------------------------------------------------
    def gap(self):
        """Distance horaire du pion 0 vers le pion 1 (sur la boucle déroulée)."""
        return START_GAP + self.pos_cum[1] - self.pos_cum[0]

    def catch_distance(self, player):
        """Combien le `player` doit avancer pour rattraper l'autre."""
        g = self.gap()
        return g if player == 0 else BOARD_N - g

    def phys_pos(self, player):
        home = 0 if player == 0 else START_GAP
        return (home + self.pos_cum[player]) % BOARD_N

    def legal_plays(self, player):
        """Coups (carte_visible, carte_cachée) jouables depuis la main.

        Les deux cartes doivent avoir des noms différents, sauf si toute la main
        porte le même nom (alors on autorise une paire identique).
        """
        hand = self.hands[player]
        names = sorted(set(hand))
        plays = []
        if len(names) >= 2:
            for up in names:
                for down in names:
                    if up != down:
                        plays.append((up, down))
        elif len(hand) >= 2:                 # main mono-nom : paire identique
            plays.append((names[0], names[0]))
        return plays

    # ---- mutations --------------------------------------------------------
    def discard(self, player, name):
        if self.discards_left[player] <= 0 or name not in self.hands[player]:
            return False
        if not self.deck:                    # paquet vide => défausse interdite
            return False
        self.hands[player].remove(name)
        self.hands[player].append(self.deck.pop())
        self.discards_left[player] -= 1
        return True

    def _refill(self, player):
        while len(self.hands[player]) < HAND_SIZE and self.deck:
            self.hands[player].append(self.deck.pop())

    def apply_turn(self, up, down, recruit_up):
        """Résout PLAY + RECRUIT pour l'actif. `recruit_up` : l'adversaire
        prend-il la carte visible ? Renvoie un dict d'infos (pour l'affichage).
        """
        active, opp = self.active, 1 - self.active
        hand = self.hands[active]
        hand.remove(up)
        hand.remove(down)

        opp_card = up if recruit_up else down
        act_card = down if recruit_up else up

        self.counts[opp][NAME_TO_ID[opp_card]] += 1
        self.counts[active][NAME_TO_ID[act_card]] += 1
        m_opp = move_for(opp_card, self.counts[opp][NAME_TO_ID[opp_card]])
        m_act = move_for(act_card, self.counts[active][NAME_TO_ID[act_card]])
        self.pos_cum[opp] += m_opp
        self.pos_cum[active] += m_act

        self._refill(active)
        self.turn += 1
        return {"active": active, "opp": opp, "opp_card": opp_card,
                "act_card": act_card, "m_opp": m_opp, "m_act": m_act}

    # ---- conditions de fin -----------------------------------------------
    def _player_status(self):
        g = self.gap()
        catch0 = g <= 0
        catch1 = g >= BOARD_N
        cb = [self.counts[0][WIN_CB] >= 3, self.counts[1][WIN_CB] >= 3]
        dd = [self.counts[0][LOSE_DD] >= 3, self.counts[1][LOSE_DD] >= 3]
        # « être en bonne posture » = remplir une condition de victoire, OU que
        # l'adversaire remplisse une condition de défaite.
        good0 = catch0 or cb[0] or dd[1]
        good1 = catch1 or cb[1] or dd[0]
        return good0, good1

    def check_end(self):
        """Renvoie le gagnant (0/1) si la partie se termine ce tour, sinon None.

        À appeler après apply_turn. Gère aussi la « panne de cartes ».
        En cas d'égalité, le joueur actif l'emporte.
        """
        active = self.active
        good0, good1 = self._player_status()
        if good0 or good1:
            if good0 and good1:
                self.winner = active            # égalité -> actif gagne
            else:
                self.winner = 0 if good0 else 1
            return self.winner

        # panne de cartes : paquet vide et l'adversaire ne pourra pas jouer 2 cartes
        opp = 1 - active
        if not self.deck and len(self.hands[opp]) < 2:
            d0, d1 = self.catch_distance(0), self.catch_distance(1)
            if d0 < d1:
                self.winner = 0
            elif d1 < d0:
                self.winner = 1
            else:
                self.winner = active            # égalité -> actif gagne
            return self.winner

        return None
