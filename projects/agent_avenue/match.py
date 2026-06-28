"""Déroulé d'une partie complète entre deux politiques (mode simple)."""

from env import AgentAvenue, BOARD_N, WIN_CB, LOSE_DD


def end_reason(env, winner):
    g = env.gap()
    if g <= 0 or g >= BOARD_N:
        return "catch"
    if env.counts[winner][WIN_CB] >= 3:
        return "codebreaker"
    if env.counts[1 - winner][LOSE_DD] >= 3:
        return "daredevil"
    return "out_of_cards"


def run_game(pol0, pol1, seed=0, max_turns=120, render=False, labels=("", "")):
    """Joue une partie. Renvoie (winner, stats, frames)."""
    env = AgentAvenue(seed=seed)
    pols = [pol0, pol1]
    frames = []

    if render:
        from render import render_frame
        frames.append(render_frame(env, None, labels))

    winner = None
    for _ in range(max_turns):
        active, opp = env.active, 1 - env.active
        if len(env.hands[active]) < 2:           # ne peut pas jouer : panne
            d0, d1 = env.catch_distance(0), env.catch_distance(1)
            winner = 0 if d0 < d1 else (1 if d1 < d0 else active)
            env.winner = winner
            break

        up, down = pols[active].choose_play(env)
        recruit_up = pols[opp].choose_recruit(env, up)
        info = env.apply_turn(up, down, recruit_up)

        if render:
            from render import render_frame
            frames.append(render_frame(env, info, labels))

        winner = env.check_end()
        if winner is not None:
            break
        env.active = opp

    stats = {"turns": env.turn, "winner": winner,
             "reason": end_reason(env, winner) if winner is not None else "none"}
    return winner, stats, frames
