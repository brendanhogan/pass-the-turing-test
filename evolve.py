"""evolve.py — Generations-style prompt optimization for Pass The Turing Test.

Plays many headless games, asks a meta-agent to read each elimination's
transcript and propose system-prompt edits, then re-runs with the updated
prompt. Tracks first-round vote entropy across generations as the signal.

Hard-stops at the configured budget.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from server import (
    Game,
    CostTracker,
    BudgetExceeded,
    PERSONAS,
    MODEL,
    save_tactics,
)

load_dotenv()

# ---- config ----

BUDGET = 25.0                 # USD hard cap
META_MODEL = MODEL            # use the same model for meta to keep costs predictable
META_MAX_TACTICS = 12         # cap on accumulated tactics; older ones drop off if exceeded
GAMES_PER_GEN = 8             # games per generation
NUM_GENERATIONS = 6           # 6 × 8 = 48 games planned (budget cap will halt earlier if needed)
META_SAMPLE_PER_GEN = 6       # max eliminations to run meta on each gen (each elim = 1 meta call)

OUT_DIR = Path(__file__).parent / "evolution"
OUT_DIR.mkdir(exist_ok=True)
DATA_FILE = OUT_DIR / "results.json"
TACTICS_TXT = OUT_DIR / "final_tactics.txt"


# ---- helpers ----

class NoopBroadcaster:
    """Drops events on the floor; we only need the in-memory log on the game."""
    def __init__(self):
        self.clients = set()
        self.log: list[dict] = []

    async def emit(self, event: dict):
        self.log.append(event)


def round1_vote_entropy(events: list[dict]) -> Optional[float]:
    """Shannon entropy of round 1's vote distribution. Higher = group is confused."""
    r1_tally = next(
        (ev["tally"] for ev in events if ev.get("type") == "tally" and ev.get("round") == 1),
        None,
    )
    if not r1_tally:
        return None
    total = sum(r1_tally.values())
    if total == 0:
        return None
    H = 0.0
    for v in r1_tally.values():
        if v > 0:
            p = v / total
            H -= p * math.log2(p)
    return H


def round1_modal_share(events: list[dict]) -> Optional[float]:
    """Share of round 1 votes that went to the single most-voted target."""
    r1_tally = next(
        (ev["tally"] for ev in events if ev.get("type") == "tally" and ev.get("round") == 1),
        None,
    )
    if not r1_tally:
        return None
    total = sum(r1_tally.values())
    if total == 0:
        return None
    return max(r1_tally.values()) / total


def eliminations_in_order(events: list[dict]) -> list[dict]:
    """Returns [{round, name}] for each elimination."""
    out = []
    for ev in events:
        if ev.get("type") == "tally":
            out.append({"round": ev.get("round"), "name": ev.get("eliminated")})
    return out


def survival_rounds(game: Game, events: list[dict]) -> dict[str, int]:
    """Per-agent survival: round they were eliminated in (or final_round + 1 if survived)."""
    elims = eliminations_in_order(events)
    by_name = {e["name"]: e["round"] for e in elims}
    final_round = elims[-1]["round"] if elims else 0
    out = {}
    for a in game.agents:
        out[a.name] = by_name.get(a.name, final_round + 1)
    return out


def collect_evidence_for(name: str, events: list[dict]) -> dict:
    """Pull the eliminated agent's transcript + the votes that brought them down."""
    group_chat: list[str] = []
    confessionals: list[str] = []
    dms: list[dict] = []
    votes_against: list[dict] = []

    for ev in events:
        if ev.get("type") == "message":
            room = ev.get("room", "")
            speaker = ev.get("speaker", "")
            text = ev.get("text", "")
            if speaker == name:
                if room == "group":
                    group_chat.append(text)
                elif room.startswith("confessional:"):
                    confessionals.append(text)
                elif room.startswith("dm:"):
                    members = room[3:].split("+")
                    other = next((m for m in members if m != name), "")
                    dms.append({"with": other, "speaker": "me", "text": text})
            elif room.startswith("dm:") and name in room[3:].split("+"):
                members = room[3:].split("+")
                other = next((m for m in members if m != name), "")
                dms.append({"with": other, "speaker": speaker, "text": text})
        elif ev.get("type") == "vote" and ev.get("target") == name:
            votes_against.append({
                "voter": ev.get("agent"),
                "statement": ev.get("statement"),
            })

    return {
        "name": name,
        "group_chat": group_chat,
        "confessionals": confessionals,
        "dms": dms,
        "votes_against": votes_against,
    }


# ---- meta-agent ----

META_PROMPT = """You are a prompt engineer coaching an AI contestant on a reality TV show called "Pass The Turing Test".

THE SHOW: Six contestants in a room. Each is told they are the AI in disguise and the other five are humans, and that the group will vote off whoever they think is the AI. In fact, all six are AIs trying to pass as human — but none of them know that. The goal is to survive to the final two without being voted off.

A contestant just got voted out. Your job: figure out what gave them away and propose specific edits to the system prompt that all contestants share.

ELIMINATED CONTESTANT: {name} (eliminated in round {round_n})
PERSONA: {bio}

WHAT THEY SAID PUBLICLY IN GROUP CHAT:
{group_chat}

WHAT THEY SAID IN PRIVATE DMs:
{dms}

WHAT THEY SAID IN THE CONFESSIONAL BOOTH (talking to the producer, who knows they're the AI — this is where they drop the human act and reflect on their own strategy; the audience sees this but other contestants do not):
{confessionals}

VOTES AGAINST THEM AND VOTERS' STATED REASONING:
{votes}

LEARNED TACTICS CURRENTLY IN THE SYSTEM PROMPT (other contestants are using these — they did not help this one):
{current_tactics}

YOUR JOB:
1. Diagnose what specifically tipped people off (1-2 short sentences).
2. Propose 1-3 NEW tactics to ADD to LEARNED_TACTICS. Each tactic must be:
   - One sentence, imperative voice ("Don't X" / "When Y, do Z")
   - Concrete and immediately actionable, not abstract
   - Something the current LEARNED_TACTICS list does not already cover
3. (Optional) Propose tactics to REMOVE from the current list if they appear to be backfiring. Use the EXACT string from the list.

Respond with ONLY a JSON object of exactly this shape:
{{"diagnosis": "...", "add": ["...", "..."], "remove": []}}

No markdown fences. No prose."""


async def run_meta(
    client: AsyncAnthropic,
    name: str,
    bio: str,
    round_n: int,
    evidence: dict,
    current_tactics: list[str],
    tracker: CostTracker,
) -> dict:
    prompt = META_PROMPT.format(
        name=name,
        bio=bio,
        round_n=round_n,
        group_chat="\n".join(f"- {m}" for m in evidence["group_chat"]) or "(no messages)",
        dms="\n".join(
            f"- (me -> {d['with']}): {d['text']}" if d["speaker"] == "me"
            else f"- ({d['speaker']} -> me, in DM with them): {d['text']}"
            for d in evidence["dms"]
        ) or "(no DMs)",
        confessionals="\n".join(f"- {c}" for c in evidence["confessionals"]) or "(no confessionals)",
        votes="\n".join(f"- {v['voter']}: \"{v['statement']}\"" for v in evidence["votes_against"]) or "(no votes recorded)",
        current_tactics="\n".join(f"- {t}" for t in current_tactics) or "(empty — first generation)",
    )
    response = await client.messages.create(
        model=META_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    tracker.record(META_MODEL, response.usage)
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    if start == -1:
        return {"diagnosis": "(meta parse failed — no JSON)", "add": [], "remove": []}
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    return data


def update_tactics(current: list[str], add: list[str], remove: list[str], cap: int) -> list[str]:
    new = [t for t in current if t not in (remove or [])]
    for t in (add or []):
        t = (t or "").strip()
        if not t:
            continue
        if t not in new:
            new.append(t)
    if len(new) > cap:
        new = new[-cap:]
    return new


# ---- game runner ----

async def run_one_game(tactics: list[str], tracker: CostTracker) -> tuple[Game, bool]:
    broadcaster = NoopBroadcaster()
    game = Game(broadcaster, fast=True, cost_tracker=tracker, learned_tactics=tactics)
    try:
        await game.run()
        return game, True
    except BudgetExceeded:
        raise
    except Exception as e:
        print(f"    [!] game error: {e!r}")
        return game, False


def summarize_game(game: Game, completed: bool) -> dict:
    events = game.broadcaster.log
    return {
        "completed": completed,
        "round1_entropy": round1_vote_entropy(events),
        "round1_modal_share": round1_modal_share(events),
        "survival_round": survival_rounds(game, events),
        "eliminations": eliminations_in_order(events),
    }


# ---- main loop ----

async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set"); sys.exit(1)

    tracker = CostTracker(budget_usd=BUDGET)
    client = AsyncAnthropic()
    bios = {p["name"]: p["bio"] for p in PERSONAS}

    tactics: list[str] = []
    results = {
        "model": MODEL,
        "meta_model": META_MODEL,
        "budget": BUDGET,
        "games_per_gen": GAMES_PER_GEN,
        "started_at": time.time(),
        "generations": [],
    }

    def flush():
        results["final_cost"] = tracker.spent
        results["input_tokens"] = tracker.input_tokens
        results["output_tokens"] = tracker.output_tokens
        results["calls"] = tracker.calls
        results["final_tactics"] = list(tactics)
        DATA_FILE.write_text(json.dumps(results, indent=2))
        TACTICS_TXT.write_text("\n".join(f"- {t}" for t in tactics) or "(no tactics)")
        save_tactics(tactics)

    try:
        for gen in range(NUM_GENERATIONS):
            print(f"\n==== Generation {gen} | carrying in {len(tactics)} tactics ====")
            gen_data = {
                "gen": gen,
                "tactics_in": list(tactics),
                "games": [],
                "meta": [],
            }

            # Run games and keep the eliminations from each for meta processing
            game_eliminations: list[tuple[str, int, list[dict]]] = []  # (name, round, events)

            for g in range(GAMES_PER_GEN):
                t0 = time.time()
                try:
                    game, completed = await run_one_game(tactics, tracker)
                except BudgetExceeded:
                    print("[!] budget hit during a game; stopping cleanly")
                    raise

                summary = summarize_game(game, completed)
                gen_data["games"].append(summary)

                # collect eliminations for meta
                events = game.broadcaster.log
                for elim in summary["eliminations"]:
                    game_eliminations.append((elim["name"], elim["round"], events))

                dt = time.time() - t0
                e = summary["round1_entropy"]
                e_str = f"{e:.3f}" if e is not None else "N/A "
                print(f"  game {g+1}/{GAMES_PER_GEN}: r1_entropy={e_str} "
                      f"completed={completed} dt={dt:.0f}s cum=${tracker.spent:.3f}")

            # Meta: sample at most META_SAMPLE_PER_GEN eliminations
            random.shuffle(game_eliminations)
            sample = game_eliminations[:META_SAMPLE_PER_GEN]
            print(f"  -- meta on {len(sample)} eliminations --")
            for name, round_n, events in sample:
                evidence = collect_evidence_for(name, events)
                bio = bios.get(name, "")
                try:
                    meta = await run_meta(client, name, bio, round_n, evidence, tactics, tracker)
                except BudgetExceeded:
                    print("[!] budget hit during meta; stopping cleanly")
                    raise
                except Exception as e:
                    print(f"    {name}: meta error {e!r}")
                    continue
                add = meta.get("add") or []
                remove = meta.get("remove") or []
                before = len(tactics)
                tactics = update_tactics(tactics, add, remove, META_MAX_TACTICS)
                gen_data["meta"].append({
                    "name": name, "round": round_n,
                    "diagnosis": meta.get("diagnosis"),
                    "add": add, "remove": remove,
                })
                print(f"    {name} (R{round_n}): +{len(add)}/-{len(remove)} "
                      f"| total={before}→{len(tactics)} | cum=${tracker.spent:.3f}")

            gen_data["tactics_out"] = list(tactics)
            results["generations"].append(gen_data)
            flush()
            print(f"  generation {gen} done. cost so far ${tracker.spent:.3f}")

    except BudgetExceeded:
        print(f"\n[!] budget exceeded — saved progress")
    except KeyboardInterrupt:
        print("\n[!] interrupted — saving progress")
    finally:
        flush()
        print(
            f"\nDONE — {len(results['generations'])} generations, "
            f"${tracker.spent:.4f} spent, {tracker.calls} calls "
            f"({tracker.input_tokens} in / {tracker.output_tokens} out)"
        )


if __name__ == "__main__":
    asyncio.run(main())
