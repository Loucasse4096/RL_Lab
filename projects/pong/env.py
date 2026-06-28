"""Environnement Pong minimaliste, sans écran (headless).

Le terrain est une grille continue : la balle a une position et une vitesse
flottantes, les deux raquettes (gauche / droite) sont des segments verticaux.
L'environnement ne connait rien au RL : il expose juste un `step(action_left,
action_right)`. Les actions sont 0 = ne rien faire, 1 = monter, 2 = descendre.

Tout est volontairement petit pour que le Q-learning tabulaire converge vite et
que le rendu reste lisible.
"""

import numpy as np

# --- Dimensions du terrain (en unités de grille) ---
WIDTH = 32.0
HEIGHT = 24.0

PADDLE_HALF = 2.5          # demi-hauteur d'une raquette
PADDLE_SPEED = 1.0         # vitesse verticale d'une raquette par pas
BALL_RADIUS = 0.6
BALL_VX = 1.0              # vitesse horizontale (constante en valeur absolue)
MAX_VY = 1.0               # vitesse verticale maximale de la balle
SPIN = 0.35               # effet ajouté selon l'endroit où la balle touche

PADDLE_X_LEFT = 1.0
PADDLE_X_RIGHT = WIDTH - 1.0

# Codes de retour de step()
HIT_LEFT = "hit_left"
HIT_RIGHT = "hit_right"
MISS_LEFT = "miss_left"      # la gauche a encaissé -> point pour la droite
MISS_RIGHT = "miss_right"    # la droite a encaissé -> point pour la gauche


class PongEnv:
    """Un Pong à deux raquettes. Une « manche » (rally) dure jusqu'à un point."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, serve_to_right=None):
        """Replace la balle au centre et relance une manche."""
        self.left_y = HEIGHT / 2.0
        self.right_y = HEIGHT / 2.0
        self.ball_x = WIDTH / 2.0
        self.ball_y = HEIGHT / 2.0
        if serve_to_right is None:
            serve_to_right = bool(self.rng.integers(0, 2))
        self.ball_vx = BALL_VX if serve_to_right else -BALL_VX
        self.ball_vy = float(self.rng.uniform(-MAX_VY, MAX_VY))
        return self

    def _move_paddle(self, y, action):
        if action == 1:
            y -= PADDLE_SPEED
        elif action == 2:
            y += PADDLE_SPEED
        return float(np.clip(y, PADDLE_HALF, HEIGHT - PADDLE_HALF))

    def step(self, action_left, action_right):
        """Avance la simulation d'un pas.

        Retourne `event` (une des constantes HIT_*/MISS_* ou None) qui décrit ce
        qui s'est passé pendant ce pas. Une manche se termine sur un MISS_*.
        """
        self.left_y = self._move_paddle(self.left_y, action_left)
        self.right_y = self._move_paddle(self.right_y, action_right)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Rebonds haut / bas
        if self.ball_y < BALL_RADIUS:
            self.ball_y = BALL_RADIUS + (BALL_RADIUS - self.ball_y)
            self.ball_vy = -self.ball_vy
        elif self.ball_y > HEIGHT - BALL_RADIUS:
            self.ball_y = (HEIGHT - BALL_RADIUS) - (self.ball_y - (HEIGHT - BALL_RADIUS))
            self.ball_vy = -self.ball_vy

        event = None

        # Raquette gauche
        if self.ball_vx < 0 and self.ball_x <= PADDLE_X_LEFT + BALL_RADIUS:
            if abs(self.ball_y - self.left_y) <= PADDLE_HALF + BALL_RADIUS:
                self.ball_x = PADDLE_X_LEFT + BALL_RADIUS
                self.ball_vx = abs(self.ball_vx)
                self.ball_vy = float(np.clip(
                    self.ball_vy + SPIN * (self.ball_y - self.left_y), -MAX_VY, MAX_VY))
                event = HIT_LEFT
            elif self.ball_x < 0:
                event = MISS_LEFT

        # Raquette droite
        elif self.ball_vx > 0 and self.ball_x >= PADDLE_X_RIGHT - BALL_RADIUS:
            if abs(self.ball_y - self.right_y) <= PADDLE_HALF + BALL_RADIUS:
                self.ball_x = PADDLE_X_RIGHT - BALL_RADIUS
                self.ball_vx = -abs(self.ball_vx)
                self.ball_vy = float(np.clip(
                    self.ball_vy + SPIN * (self.ball_y - self.right_y), -MAX_VY, MAX_VY))
                event = HIT_RIGHT
            elif self.ball_x > WIDTH:
                event = MISS_RIGHT

        return event
