"""Agent Q-learning tabulaire pour le Pong.

Astuce clé : on n'apprend qu'UNE seule table Q, du point de vue d'une raquette
« à droite ». La raquette de gauche réutilise la même table en regardant le
terrain en miroir. Résultat : l'agent joue contre lui-même (self-play) et chaque
pas de simulation produit deux expériences d'apprentissage. Ça converge vite.

Un état est encodé en un seul entier (index dans la table) à partir de :
  - rel  : position verticale de la balle par rapport à la raquette (17 paliers)
  - bx   : distance horizontale balle <-> raquette (8 paliers)
  - vers : la balle vient-elle vers nous ? (2)
  - vy   : la balle descend-elle ? (2)
=> 17 * 8 * 2 * 2 = 544 états, 3 actions.
"""

import numpy as np

from env import WIDTH, HEIGHT, PADDLE_X_RIGHT

REL_BINS = 8          # rel clampé dans [-8, 8] -> 17 valeurs
N_REL = 2 * REL_BINS + 1
N_BX = 8
N_STATES = N_REL * N_BX * 2 * 2
N_ACTIONS = 3


def _egocentric(ball_x, ball_y, ball_vx, ball_vy, paddle_y, side):
    """Renvoie une vue « comme si j'étais la raquette de droite ».

    Pour la gauche, on applique un miroir horizontal du terrain.
    """
    if side == "left":
        ball_x = WIDTH - ball_x
        ball_vx = -ball_vx
    # distance horizontale jusqu'à notre raquette (à droite après miroir)
    dist_x = PADDLE_X_RIGHT - ball_x
    approaching = ball_vx > 0          # va vers notre raquette
    return ball_y, ball_vy, paddle_y, dist_x, approaching


def encode(ball_x, ball_y, ball_vx, ball_vy, paddle_y, side):
    ball_y, ball_vy, paddle_y, dist_x, approaching = _egocentric(
        ball_x, ball_y, ball_vx, ball_vy, paddle_y, side)

    rel = int(round(ball_y - paddle_y))
    rel = max(-REL_BINS, min(REL_BINS, rel)) + REL_BINS      # 0..16
    bx = int(max(0.0, dist_x) / (WIDTH / N_BX))
    bx = max(0, min(N_BX - 1, bx))
    vers = 1 if approaching else 0
    vy = 1 if ball_vy > 0 else 0

    return ((rel * N_BX + bx) * 2 + vers) * 2 + vy


class QAgent:
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.2, seed=0):
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def update(self, s, a, r, s_next, terminal):
        target = r if terminal else r + self.gamma * float(self.q[s_next].max())
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def save(self, path):
        np.save(path, self.q)

    @classmethod
    def load(cls, path):
        agent = cls()
        agent.q = np.load(path)
        agent.epsilon = 0.0
        return agent
