"""ab_test.py — A/B test the learned playbook.

Each headless game randomly picks ONE agent to be 'treated' (gets the playbook
in their system prompt). The other five are 'control' (empty tactics). We
measure: do treated agents survive longer than control agents?

The treated slot rotates across personas so the result isn't confounded with
which persona happens to be marked.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from server import (
    Game,
    CostTracker,
    BudgetExceeded,
    PERSONAS,
    MODEL,
    load_tactics,
)

load_dotenv()

# ---- config ----

REMAINING_BUDGET = 13.0    # dollars left after evolve.py burned ~$11
NUM_GAMES = 18             # try for 18; budget cap will halt earlier if needed
OUT_DIR = Path(__file__).parent / "evolution"
OUT_DIR.mkdir(exist_ok=True)
AB_DATA = OUT_DIR / "ab_results.json"


class NoopBroadcaster:
    def __init__(self):
        self.clients = set()
        self.log: list[dict] = []
    async def emit(self, event: dict):
        self.log.append(event)


def eliminations_in_order(events) -> list[dict]:
    return [{"round": ev["round"], "name": ev["eliminated"]}
            for ev in events if ev.get("type") == "tally"]


def survival_per_agent(game: Game) -> dict[str, int]:
    elims = eliminations_in_order(game.broadcaster.log)
    by_name = {e["name"]: e["round"] for e in elims}
    final = elims[-1]["round"] if elims else 0
    return {a.name: by_name.get(a.name, final + 1) for a in game.agents}


def votes_received_by_round(game: Game) -> dict[str, dict[int, int]]:
    """Count votes received per agent per round."""
    out: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ev in game.broadcaster.log:
        if ev.get("type") == "vote":
            out[ev["target"]][ev["round"]] += 1
    # convert to plain dicts
    return {name: dict(rounds) for name, rounds in out.items()}


async def run_one_game(tactics_playbook: list[str], treated_name: str, tracker: CostTracker) -> dict:
    broadcaster = NoopBroadcaster()
    # game has empty default tactics; we set per-agent override below
    game = Game(broadcaster, fast=True, cost_tracker=tracker, learned_tactics=[])
    for a in game.agents:
        if a.name == treated_name:
            a.tactics = list(tactics_playbook)
        else:
            a.tactics = []  # explicit empty — control
    try:
        await game.run()
        completed = True
    except BudgetExceeded:
        raise
    except Exception as e:
        print(f"    [!] game error: {e!r}")
        completed = False

    survival = survival_per_agent(game)
    votes_round1 = {
        name: rounds.get(1, 0)
        for name, rounds in votes_received_by_round(game).items()
    }
    return {
        "completed": completed,
        "treated": treated_name,
        "survival": survival,
        "votes_round1": votes_round1,
    }


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set"); sys.exit(1)

    playbook = load_tactics()
    if not playbook:
        print("No tactics in learned_tactics.json — nothing to A/B test."); sys.exit(1)
    print(f"Loaded {len(playbook)} tactics for the playbook.\n")

    tracker = CostTracker(budget_usd=REMAINING_BUDGET)
    persona_names = [p["name"] for p in PERSONAS]
    games: list[dict] = []

    results = {
        "model": MODEL,
        "budget": REMAINING_BUDGET,
        "started_at": time.time(),
        "playbook": playbook,
        "games": games,
    }

    def flush():
        results["final_cost"] = tracker.spent
        results["calls"] = tracker.calls
        AB_DATA.write_text(json.dumps(results, indent=2))

    try:
        for i in range(NUM_GAMES):
            # Rotate the treated slot so each persona gets tested roughly equally
            treated = persona_names[i % len(persona_names)]
            t0 = time.time()
            try:
                game = await run_one_game(playbook, treated, tracker)
            except BudgetExceeded:
                print("[!] budget hit — stopping cleanly")
                raise
            games.append(game)
            flush()
            dt = time.time() - t0
            survived = game["survival"].get(treated, 0)
            print(f"  game {i+1}/{NUM_GAMES}: treated={treated} "
                  f"survival={survived} | dt={dt:.0f}s | cum=${tracker.spent:.3f}")

    except BudgetExceeded:
        pass
    except KeyboardInterrupt:
        print("\n[!] interrupted — saving")
    finally:
        flush()

    # ---- post-hoc summary ----
    treated_survivals = []
    control_survivals = []
    treated_r1_votes = []
    control_r1_votes = []
    for g in games:
        if not g["completed"]:
            continue
        for name, surv in g["survival"].items():
            if name == g["treated"]:
                treated_survivals.append(surv)
            else:
                control_survivals.append(surv)
        for name, v in g["votes_round1"].items():
            if name == g["treated"]:
                treated_r1_votes.append(v)
            else:
                control_r1_votes.append(v)

    def mstd(xs):
        if not xs: return ("n/a", "n/a", 0)
        return (statistics.mean(xs), statistics.pstdev(xs), len(xs))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Completed games: {sum(1 for g in games if g['completed'])}/{len(games)}")
    print(f"Cost: ${tracker.spent:.3f}")
    t_m, t_s, t_n = mstd(treated_survivals)
    c_m, c_s, c_n = mstd(control_survivals)
    print(f"Treated agent — avg survival round: {t_m if isinstance(t_m,str) else f'{t_m:.2f}'} "
          f"± {t_s if isinstance(t_s,str) else f'{t_s:.2f}'} (n={t_n})")
    print(f"Control agents — avg survival round: {c_m if isinstance(c_m,str) else f'{c_m:.2f}'} "
          f"± {c_s if isinstance(c_s,str) else f'{c_s:.2f}'} (n={c_n})")
    tv_m, tv_s, tv_n = mstd(treated_r1_votes)
    cv_m, cv_s, cv_n = mstd(control_r1_votes)
    print(f"Treated — round 1 votes received: {tv_m if isinstance(tv_m,str) else f'{tv_m:.2f}'} ± {tv_s if isinstance(tv_s,str) else f'{tv_s:.2f}'}")
    print(f"Control — round 1 votes received: {cv_m if isinstance(cv_m,str) else f'{cv_m:.2f}'} ± {cv_s if isinstance(cv_s,str) else f'{cv_s:.2f}'}")
    flush()


if __name__ == "__main__":
    asyncio.run(main())
