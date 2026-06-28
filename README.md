# 🧪 RL_Lab

Une collection de **petits projets de reinforcement learning**, simples et
surtout **très visuels** — le genre où on regarde l'ordinateur devenir bon au
fil des itérations, et où on fait s'affronter ses différentes versions.

Tout reste léger (numpy + Pillow, pas de GPU, pas de framework lourd) pour que
chaque projet soit lisible de bout en bout et s'entraîne en quelques minutes.

## Projets

| Projet | Description | Aperçu |
|--------|-------------|--------|
| [🏓 Pong](projects/pong/) | Un Pong qui apprend à jouer contre lui-même (self-play, Q-learning). On le regarde progresser et on fait s'affronter ses versions. | ![pong](projects/pong/media/progression.gif) |

## Démarrer

```bash
pip install -r requirements.txt
cd projects/pong
python train.py        # entraîne et sauvegarde des versions
python progression.py  # le GIF "regarde-le progresser"
python tournament.py 1000 20000   # un duel entre deux versions
```

## Structure

```
RL_Lab/
├── requirements.txt
└── projects/
    └── pong/          # premier projet : Pong self-play
        ├── env.py         # l'environnement (le jeu)
        ├── agent.py       # l'agent qui apprend
        ├── train.py       # l'entraînement
        ├── match.py       # un match entre deux politiques
        ├── render.py      # rendu en GIF
        ├── tournament.py  # faire s'affronter deux versions
        ├── progression.py # le GIF de progression
        ├── checkpoints/   # versions entraînées (.npy)
        └── media/         # GIFs générés
```

Chaque nouveau projet vit dans son propre dossier sous `projects/`, autonome,
avec son README.
