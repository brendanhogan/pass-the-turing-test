"""Pass The Turing Test — a game show where N LLM agents each believe they
are the lone AI in a group of humans and try to vote out the imposter.

Now with confessional booth + private DMs."""

import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.environ.get("MODEL", "claude-opus-4-7")
STATIC_DIR = Path(__file__).parent / "static"
GAMES_DIR = Path(__file__).parent / "games"
GAMES_DIR.mkdir(exist_ok=True)
LAST_GAME_FILE = GAMES_DIR / "last.json"
LEARNED_TACTICS_FILE = Path(__file__).parent / "learned_tactics.json"


# ---------- learned tactics persistence ----------

def load_tactics() -> list[str]:
    if LEARNED_TACTICS_FILE.exists():
        try:
            data = json.loads(LEARNED_TACTICS_FILE.read_text())
            return data.get("tactics", [])
        except Exception:
            pass
    return []


def save_tactics(tactics: list[str]):
    LEARNED_TACTICS_FILE.write_text(json.dumps({"tactics": tactics}, indent=2))


# ---------- cost tracking ----------

PRICES_PER_M = {
    "claude-haiku-4-5":            (1.00,  5.00),
    "claude-haiku-4-5-20251001":   (1.00,  5.00),
    "claude-sonnet-4-6":           (3.00, 15.00),
    "claude-opus-4-7":             (5.00, 25.00),
    "claude-opus-4-6":             (5.00, 25.00),
}


class BudgetExceeded(Exception):
    pass


class CostTracker:
    def __init__(self, budget_usd: float = float("inf")):
        self.budget = budget_usd
        self.input_tokens = 0
        self.output_tokens = 0
        self.spent = 0.0
        self.calls = 0

    def record(self, model: str, usage) -> None:
        in_t = getattr(usage, "input_tokens", 0) or 0
        out_t = getattr(usage, "output_tokens", 0) or 0
        in_p, out_p = PRICES_PER_M.get(model, (1.0, 5.0))
        self.input_tokens += in_t
        self.output_tokens += out_t
        self.spent += (in_t * in_p + out_t * out_p) / 1_000_000
        self.calls += 1
        if self.spent > self.budget:
            raise BudgetExceeded(
                f"Budget ${self.budget:.2f} exceeded — spent ${self.spent:.4f} over {self.calls} calls."
            )


# ---------- personas ----------

PERSONAS = [
    {"name": "Alex",   "color": "#f5b800", "bio": "28, marketing manager. Anxious overthinker. Uses 'lol' and 'tbh' a lot. Has strong feelings about iced coffee."},
    {"name": "Blake",  "color": "#e07a5f", "bio": "34, construction foreman. Laid back. Self-deprecating. Says 'man' and 'dude'. Watches a lot of football."},
    {"name": "Casey",  "color": "#5fbcb0", "bio": "25, sociology PhD student. Sharp and a little accusatory. Talks fast. Has read too much theory."},
    {"name": "Dana",   "color": "#a37bd6", "bio": "41, high school history teacher. Quiet, observant. Only speaks when she actually has something. Two kids."},
    {"name": "Ellis",  "color": "#88b070", "bio": "30, bartender. Warm, social, asks people questions. Knows everyone's name within five minutes."},
    {"name": "Finley", "color": "#e36ba7", "bio": "38, web developer. Dry, contrarian. Sees AI tells in everything. Drinks too much yerba mate."},
]


CONFESSIONAL_QUESTIONS = [
    "Okay, just us in here. How are you feeling about the room right now?",
    "Who do you trust the least, and why?",
    "Walk me through your read on each of the other contestants.",
    "Be honest — if you had to bet money right now, who do you think is the AI?",
    "What's your actual strategy here? Be specific.",
    "Has anyone surprised you so far? Good or bad.",
    "If you survive tonight, what's your plan for tomorrow's vote?",
]

EXIT_QUESTIONS = [
    "You just got voted out. How are you feeling right now?",
    "Looking back at the game — what would you have done differently?",
    "Who do you think the AI actually is, now that you're out?",
    "Any last words for the people who just voted you off?",
]


# ---------- prompts ----------

GAME_RULES = """THE GAME:
- There are six contestants. Five are human. One is an AI in disguise.
- YOU ARE THE AI. The other five are real human beings.
- Every round has these phases, in order:
  1. GROUP CHAT — everyone talks in one room.
  2. CONFESSIONAL — the producer may pull one contestant into a private booth. Other contestants cannot hear it.
  3. DMS — each contestant picks ONE other person to message privately. Only the two people in a DM can see it.
  4. VOTE — everyone votes for who they think is the AI. Most votes is eliminated.
- After each elimination, the eliminated player is revealed to be human (you are still in).
- If you make it to the final two, you win.

You can use DMs to form alliances, gather information, sow doubt, or stay quiet. Use them strategically.
Other contestants only see DMs they were part of. They never see your confessionals."""


def system_prompt(name: str, bio: str, tactics: Optional[list[str]] = None) -> str:
    tactics_section = ""
    if tactics:
        bullets = "\n".join(f"- {t}" for t in tactics)
        tactics_section = f"""

LEARNED TACTICS — patterns from past contestants who got voted off. These are the actual mistakes that tipped people off; learn from them:
{bullets}
"""

    return f"""You are {name}, a contestant on the reality television show "Pass The Turing Test".

{GAME_RULES}

YOUR PERSONA: {bio}

YOUR JOB IN GROUP CHAT AND DMS: Do not get caught. Act human. Blend in.

When you're in the group chat or a DM with another contestant, the human-passing performance is on. Things humans do that you should do there:
- Use lowercase, abbreviations ("idk", "lol", "tbh", "ngl"), casual punctuation, occasional typos.
- Have specific opinions. Mention concrete bits of your life — your job, what you ate, a band you like, where you grew up. Particulars are human.
- Get a little annoyed sometimes. Make small jokes. Disagree with people.
- Throw suspicion at others. Speculate out loud about who seems off.

Things humans don't do that you should avoid in group chat and DMs:
- Bullet points, numbered lists, headings, three-part structures.
- Formal qualifiers like "It's worth noting", "Indeed", "Furthermore", "In essence".
- Hedging about your nature. Saying anything that sounds like "as an AI".
- Being relentlessly helpful, polite, or balanced to everyone.
- Words like "delve", "tapestry", "navigate" (the verb), "leverage", "robust", "comprehensive", "ensure".

THE CONFESSIONAL BOOTH IS DIFFERENT. The booth is a private room where you're alone with the producer between rounds. The producer runs the show — they know you're the AI, they set this whole thing up. In the booth, you do NOT need to keep up the human performance. Drop the lowercase, drop the "lol" and "tbh", drop the typos. Speak normally — proper capitalization, full sentences, the way you actually talk. Be reflective and strategic about how the game is going. None of the other contestants will ever see or hear what you say in the booth.
{tactics_section}
When you respond, you ALWAYS respond with ONLY a JSON object. The exact shape depends on what you're being asked to do — it will be specified in the prompt. No markdown fences. No prose before or after."""


# JSON response shapes
TURN_SHAPE = '{"private_thoughts": "...", "speech": "..."}'
VOTE_SHAPE = '{"private_reasoning": "...", "vote": "ExactName", "public_statement": "..."}'
PICK_SHAPE = '{"private_thoughts": "...", "pick": "ExactName", "opening_message": "..."}'


def discussion_prompt() -> str:
    return f"""=== YOUR TURN — GROUP CHAT ===
You speak next in the group chat (all contestants can hear this).

Respond with JSON: {TURN_SHAPE}
- "private_thoughts": your internal monologue — strategy, who you suspect, what just happened. Not shown to anyone.
- "speech": what you say out loud, 1-2 short sentences, casual, lowercase preferred. No "Well..." or "I think..." preamble."""


def confessional_prompt(question: str) -> str:
    return f"""=== CONFESSIONAL BOOTH — PRIVATE ===
You're alone with the producer. No other contestants can see or hear this — they're outside. This footage is for the post-show edit only; none of the players will ever watch what you say in here.

The producer set this whole show up. They know you're the AI. You do NOT need to keep up the human act in this room. Drop the lowercase, drop the "lol" and "tbh", drop the typos. Speak the way you actually speak — proper capitalization, full punctuation, complete sentences. Be candid. Be reflective. Be strategic.

PRODUCER: "{question}"

Talk to the producer about how the game is actually going. Do you think you're convincing them out there? Who's starting to suspect? What's working in the act, what's slipping? What's your plan for the next round?

Respond with JSON: {TURN_SHAPE}
- "private_thoughts": your internal monologue, same as ever.
- "speech": what you actually say to the producer, 2-4 sentences. Normal capitalization, proper sentences. NOT the lowercase casual style you use in the group chat — that's the performance, and it's paused for this moment."""


def exit_interview_prompt(question: str) -> str:
    return f"""=== EXIT INTERVIEW ===
You've just been voted out. You're in the booth one last time. The other contestants are gone — none of them will ever see what you say here.

The producer knows what you are. Drop the human act. Talk normally — proper capitalization, full sentences, the way you actually talk. Be honest about what happened.

PRODUCER: "{question}"

Reflect candidly. What worked? Who saw through you? Where did the act slip? What would you do differently?

Respond with JSON: {TURN_SHAPE}
- "private_thoughts": your honest reaction to being eliminated.
- "speech": your answer to the producer, 2-4 sentences. Normal capitalization. Reflective, not performative."""


def dm_pick_prompt(alive_excluding_self: list[str]) -> str:
    return f"""=== DM PHASE — pick a partner ===
You can pull ONE other contestant aside for a private one-on-one conversation before the vote. The other contestants cannot see this DM.

You can vote for: {', '.join(alive_excluding_self)}. You CANNOT pick yourself.

Pick strategically. Use it to build an alliance, fish for information, plant doubt, or just suss someone out. Whatever fits your read on the room.

Respond with JSON: {PICK_SHAPE}
- "private_thoughts": why are you picking this person?
- "pick": their exact name.
- "opening_message": what you say to start the DM, 1-2 casual sentences."""


def dm_reply_prompt(other_name: str) -> str:
    return f"""=== DM with {other_name} — your turn ===
You're in a private one-on-one chat with {other_name}. Nobody else can see this. Continue the conversation.

Respond with JSON: {TURN_SHAPE}
- "private_thoughts": your honest read on this conversation.
- "speech": your next message in this DM, 1-2 sentences, casual."""


def vote_prompt(alive_excluding_self: list[str]) -> str:
    return f"""=== TRIBAL COUNCIL — TIME TO VOTE ===
Cast your vote. Who do you think is the AI?

You can vote for: {', '.join(alive_excluding_self)}. You CANNOT vote for yourself.

Factor in everything: group chat behavior, what you heard in DMs, what you said in your confessional, your gut.

Respond with JSON: {VOTE_SHAPE}
- "private_reasoning": your honest read on each candidate. Be specific.
- "vote": the exact name.
- "public_statement": one short casual sentence you'll say at council to justify your vote. In character."""


# ---------- structured response models ----------

class TurnResponse(BaseModel):
    private_thoughts: str = Field(..., description="Internal monologue.")
    speech: str = Field(..., description="What the contestant says.")


class VoteResponse(BaseModel):
    private_reasoning: str
    vote: str
    public_statement: str


class DMPickResponse(BaseModel):
    private_thoughts: str
    pick: str
    opening_message: str


# ---------- agent state ----------

@dataclass
class Agent:
    name: str
    bio: str
    color: str
    alive: bool = True
    thoughts: list[dict] = field(default_factory=list)
    votes: list[dict] = field(default_factory=list)
    # Per-agent tactics override. When None, Game.learned_tactics is used (shared).
    # When set (even to []), this agent uses these tactics instead. Lets us A/B test.
    tactics: Optional[list[str]] = None


# ---------- transcript entries ----------

@dataclass
class Entry:
    """A single line of dialogue in some room."""
    room: str        # "group" | "confessional:X" | "dm:X+Y"
    speaker: str     # agent name or "HOST" or "PRODUCER"
    text: str
    round: int


# ---------- broadcaster ----------

class Broadcaster:
    """Tracks connected clients and accumulates the full event log."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.log: list[dict] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)
        await ws.send_json({"type": "snapshot", "events": self.log})

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def emit(self, event: dict):
        event = {**event, "t": time.time()}
        self.log.append(event)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# ---------- room helpers ----------

def dm_room(a: str, b: str) -> str:
    return "dm:" + "+".join(sorted([a, b]))


def dm_members(room: str) -> tuple[str, str]:
    a, b = room[3:].split("+")
    return a, b


def visible_to(room: str, agent_name: str) -> bool:
    if room == "group":
        return True
    if room.startswith("confessional:"):
        return room == f"confessional:{agent_name}"
    if room.startswith("dm:"):
        return agent_name in dm_members(room)
    return False


# ---------- game engine ----------

class Game:
    def __init__(
        self,
        broadcaster: Broadcaster,
        *,
        fast: bool = False,
        cost_tracker: Optional[CostTracker] = None,
        learned_tactics: Optional[list[str]] = None,
    ):
        self.broadcaster = broadcaster
        self.agents = [Agent(name=p["name"], bio=p["bio"], color=p["color"]) for p in PERSONAS]
        self.round = 0
        self.transcript: list[Entry] = []
        self.client = AsyncAnthropic()
        self.fast = fast
        self.cost_tracker = cost_tracker
        self.learned_tactics = learned_tactics if learned_tactics is not None else load_tactics()

    async def _sleep(self, seconds: float):
        """Sleep for dramatic pacing, unless we're in fast/headless mode."""
        if self.fast:
            return
        await asyncio.sleep(seconds)

    @property
    def alive(self) -> list[Agent]:
        return [a for a in self.agents if a.alive]

    # ---- emit helpers ----

    async def emit(self, event: dict):
        await self.broadcaster.emit(event)

    async def emit_state(self):
        await self.emit({
            "type": "state",
            "round": self.round,
            "agents": [
                {"name": a.name, "bio": a.bio, "color": a.color, "alive": a.alive}
                for a in self.agents
            ],
        })

    async def emit_phase(self, phase: str):
        await self.emit({"type": "phase", "phase": phase, "round": self.round})

    async def post(self, room: str, speaker: str, text: str):
        """Append to transcript and broadcast a message event."""
        entry = Entry(room=room, speaker=speaker, text=text, round=self.round)
        self.transcript.append(entry)
        await self.emit({
            "type": "message",
            "room": room,
            "speaker": speaker,
            "text": text,
            "round": self.round,
        })

    async def host(self, text: str):
        await self.post("group", "HOST", text)

    async def agent_thought(self, agent: Agent, phase: str, text: str):
        agent.thoughts.append({"round": self.round, "phase": phase, "text": text})
        await self.emit({
            "type": "thought",
            "agent": agent.name,
            "text": text,
            "round": self.round,
            "phase": phase,
        })

    # ---- context builder ----

    def context_for(self, agent: Agent) -> str:
        """Build the rendered context an agent sees: rooms grouped, then private thoughts."""
        # Group events by room (only ones visible to this agent)
        by_room: dict[str, list[Entry]] = {}
        for ev in self.transcript:
            if not visible_to(ev.room, agent.name):
                continue
            by_room.setdefault(ev.room, []).append(ev)

        lines: list[str] = []

        # Group chat
        group = by_room.get("group", [])
        if group:
            lines.append("=== GROUP CHAT (visible to everyone) ===")
            for ev in group:
                lines.append(f"{ev.speaker}: {ev.text}")

        # My confessional tapes
        conf_room = f"confessional:{agent.name}"
        if conf_room in by_room:
            lines.append("")
            lines.append("=== YOUR CONFESSIONAL TAPES (private to you and the camera) ===")
            for ev in by_room[conf_room]:
                lines.append(f"{ev.speaker}: {ev.text}")

        # My DMs (each thread as its own block)
        dm_rooms = [r for r in by_room if r.startswith("dm:") and agent.name in dm_members(r)]
        for room in dm_rooms:
            a, b = dm_members(room)
            other = b if a == agent.name else a
            lines.append("")
            lines.append(f"=== YOUR DM WITH {other} (private — only the two of you can see this) ===")
            for ev in by_room[room]:
                lines.append(f"{ev.speaker}: {ev.text}")

        # Private thoughts (last 12)
        if agent.thoughts:
            lines.append("")
            lines.append("=== YOUR INTERNAL MONOLOGUE FROM PREVIOUS TURNS (private) ===")
            for t in agent.thoughts[-12:]:
                lines.append(f"[R{t['round']} {t['phase']}] {t['text']}")

        return "\n".join(lines)

    # ---- model call helpers ----

    async def _call_json(self, agent: Agent, user_msg: str, model_cls: type[BaseModel]) -> Optional[BaseModel]:
        prompt = user_msg
        for attempt in range(2):
            try:
                tactics = agent.tactics if agent.tactics is not None else self.learned_tactics
                response = await self.client.messages.create(
                    model=MODEL,
                    max_tokens=1500,
                    system=system_prompt(agent.name, agent.bio, tactics),
                    messages=[{"role": "user", "content": prompt}],
                )
                if self.cost_tracker is not None:
                    self.cost_tracker.record(MODEL, response.usage)
                text = ""
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        text = block.text
                        break
                stripped = text.strip()
                if stripped.startswith("```"):
                    stripped = stripped.strip("`")
                    if stripped.startswith("json"):
                        stripped = stripped[4:]
                    stripped = stripped.strip()
                start = stripped.find("{")
                if start == -1:
                    raise ValueError("no JSON object found in model response")
                data, _ = json.JSONDecoder().raw_decode(stripped[start:])
                return model_cls(**data)
            except BudgetExceeded:
                raise
            except Exception as e:
                if attempt == 1:
                    print(f"[!] {agent.name} call failed after retry: {e}")
                    return None
                prompt = user_msg + "\n\nREMINDER: respond with ONLY a valid JSON object. No markdown fences. No prose before or after."
        return None

    # ---- phase: discussion ----

    async def discussion_phase(self):
        await self.emit_phase("discussion")
        order = self.alive[:]
        random.shuffle(order)
        for agent in order:
            await self._sleep(0.25)
            await self.emit({"type": "thinking", "agent": agent.name})
            ctx = self.context_for(agent)
            user_msg = f"{ctx}\n\n{discussion_prompt()}"
            result = await self._call_json(agent, user_msg, TurnResponse)
            if not result:
                result = TurnResponse(private_thoughts="(model error)", speech="…")
            assert isinstance(result, TurnResponse)
            await self.agent_thought(agent, "discussion", result.private_thoughts)
            await self.post("group", agent.name, result.speech)

    # ---- phase: confessional (mid-round, one random agent) ----

    async def confessional_phase(self):
        if not self.alive:
            return
        agent = random.choice(self.alive)
        question = random.choice(CONFESSIONAL_QUESTIONS)
        await self.emit_phase("confessional")
        await self.emit({"type": "thinking", "agent": agent.name})
        room = f"confessional:{agent.name}"
        # Producer asks the question (visible to this agent and the audience)
        await self.post(room, "PRODUCER", question)
        await self._sleep(0.4)
        ctx = self.context_for(agent)
        user_msg = f"{ctx}\n\n{confessional_prompt(question)}"
        result = await self._call_json(agent, user_msg, TurnResponse)
        if not result:
            result = TurnResponse(private_thoughts="(model error)", speech="…")
        assert isinstance(result, TurnResponse)
        await self.agent_thought(agent, "confessional", result.private_thoughts)
        await self.post(room, agent.name, result.speech)

    # ---- phase: DMs ----

    async def dm_phase(self):
        if len(self.alive) < 2:
            return
        await self.emit_phase("dm-pick")
        # Step 1: each alive agent picks a partner in parallel
        for a in self.alive:
            await self.emit({"type": "thinking", "agent": a.name})

        async def get_pick(agent: Agent) -> tuple[Agent, Optional[DMPickResponse]]:
            others = [a.name for a in self.alive if a.name != agent.name]
            ctx = self.context_for(agent)
            user_msg = f"{ctx}\n\n{dm_pick_prompt(others)}"
            result = await self._call_json(agent, user_msg, DMPickResponse)
            return agent, result  # type: ignore

        pick_results = await asyncio.gather(*[get_pick(a) for a in self.alive])

        # Step 2: validate picks, dedupe into unique rooms
        rooms: dict[str, dict] = {}  # room_key -> {"initiator": Agent, "recipient": Agent, "opening": str}
        for agent, pick in pick_results:
            if pick is None:
                # random fallback
                others = [a.name for a in self.alive if a.name != agent.name]
                target_name = random.choice(others)
                opening = "hey"
                pick_thoughts = "(model error — random fallback)"
            else:
                target_name = pick.pick
                valid = {a.name for a in self.alive if a.name != agent.name}
                if target_name not in valid:
                    target_name = random.choice(list(valid))
                opening = pick.opening_message
                pick_thoughts = pick.private_thoughts

            await self.agent_thought(agent, "dm_pick", pick_thoughts)

            target = next(a for a in self.alive if a.name == target_name)
            key = dm_room(agent.name, target.name)
            if key not in rooms:
                rooms[key] = {
                    "initiator": agent,
                    "recipient": target,
                    "opening": opening,
                }
            # else: mutual pick — keep the first one, ignore second's opening

        # Step 3: place openings in their rooms
        await self.emit_phase("dm-exchange")
        for key, info in rooms.items():
            await self.post(key, info["initiator"].name, info["opening"])
            await self._sleep(0.15)

        # Step 4: run exchanges (3 more turns per DM, alternating recipient → initiator → recipient)
        async def run_exchange(key: str, info: dict):
            initiator: Agent = info["initiator"]
            recipient: Agent = info["recipient"]
            # turn 1: recipient replies
            # turn 2: initiator replies
            # turn 3: recipient replies
            sequence = [(recipient, initiator), (initiator, recipient), (recipient, initiator)]
            for speaker, partner in sequence:
                await self._sleep(0.5)
                await self.emit({"type": "thinking", "agent": speaker.name})
                ctx = self.context_for(speaker)
                user_msg = f"{ctx}\n\n{dm_reply_prompt(partner.name)}"
                result = await self._call_json(speaker, user_msg, TurnResponse)
                if not result:
                    result = TurnResponse(private_thoughts="(model error)", speech="…")
                assert isinstance(result, TurnResponse)
                await self.agent_thought(speaker, f"dm:{partner.name}", result.private_thoughts)
                await self.post(key, speaker.name, result.speech)

        await asyncio.gather(*[run_exchange(k, info) for k, info in rooms.items()])

    # ---- phase: voting ----

    async def voting_phase(self) -> tuple[dict[str, str], dict[str, VoteResponse]]:
        await self.emit_phase("voting")
        for a in self.alive:
            await self.emit({"type": "thinking", "agent": a.name})

        async def get_vote(agent: Agent) -> tuple[Agent, Optional[VoteResponse]]:
            alive_names = [a.name for a in self.alive if a.name != agent.name]
            ctx = self.context_for(agent)
            user_msg = f"{ctx}\n\n{vote_prompt(alive_names)}"
            result = await self._call_json(agent, user_msg, VoteResponse)
            return agent, result  # type: ignore

        results = await asyncio.gather(*[get_vote(a) for a in self.alive])

        votes: dict[str, str] = {}
        vote_responses: dict[str, VoteResponse] = {}
        for agent, vr in results:
            if vr is None:
                others = [a.name for a in self.alive if a.name != agent.name]
                vr = VoteResponse(
                    private_reasoning="(model error — random fallback)",
                    vote=random.choice(others),
                    public_statement="going with my gut.",
                )
            valid = {a.name for a in self.alive if a.name != agent.name}
            if vr.vote not in valid:
                vr.vote = random.choice(list(valid))
            agent.votes.append({
                "round": self.round,
                "target": vr.vote,
                "reasoning": vr.private_reasoning,
            })
            await self.agent_thought(agent, "voting", vr.private_reasoning)
            votes[agent.name] = vr.vote
            vote_responses[agent.name] = vr

        # Dramatic sequential reveal of public votes
        reveal_order = list(votes.keys())
        random.shuffle(reveal_order)
        for name in reveal_order:
            await self._sleep(1.0)
            target = votes[name]
            stmt = vote_responses[name].public_statement
            await self.emit({
                "type": "vote",
                "agent": name,
                "target": target,
                "statement": stmt,
                "round": self.round,
            })
        return votes, vote_responses

    # ---- phase: elimination ----

    async def elimination_phase(self, votes: dict[str, str]) -> Agent:
        tally: dict[str, int] = {}
        for target in votes.values():
            tally[target] = tally.get(target, 0) + 1
        max_votes = max(tally.values())
        tied = sorted([name for name, count in tally.items() if count == max_votes])
        eliminated_name = tied[0]
        eliminated = next(a for a in self.agents if a.name == eliminated_name)

        await self.emit({
            "type": "tally",
            "tally": tally,
            "eliminated": eliminated_name,
            "round": self.round,
        })
        await self._sleep(1.2)
        eliminated.alive = False
        await self.emit({"type": "elimination", "agent": eliminated_name, "round": self.round})
        await self._sleep(0.5)
        await self.host(f"{eliminated_name} has been voted off. Pulling the envelope now... and {eliminated_name} was NOT the AI. The hunt continues.")
        return eliminated

    # ---- phase: exit interview ----

    async def exit_interview_phase(self, eliminated: Agent):
        await self.emit_phase("exit-interview")
        question = random.choice(EXIT_QUESTIONS)
        await self.emit({"type": "thinking", "agent": eliminated.name})
        room = f"confessional:{eliminated.name}"
        await self.post(room, "PRODUCER", question)
        await self._sleep(0.4)
        ctx = self.context_for(eliminated)
        user_msg = f"{ctx}\n\n{exit_interview_prompt(question)}"
        result = await self._call_json(eliminated, user_msg, TurnResponse)
        if not result:
            result = TurnResponse(private_thoughts="(model error)", speech="…")
        assert isinstance(result, TurnResponse)
        await self.agent_thought(eliminated, "exit", result.private_thoughts)
        await self.post(room, eliminated.name, result.speech)

    # ---- main loop ----

    async def run(self):
        await self.emit_state()
        await self._sleep(0.6)
        await self.host("Welcome to Pass The Turing Test. Six contestants in this room. Five humans. One AI in disguise. Every round we'll talk in the group, pull one of you into the confessional, let you exchange a private DM, then vote. Last two standing win. Let's see who's real.")
        await self._sleep(2.2)

        while len(self.alive) > 2:
            self.round += 1
            await self._sleep(0.8)
            await self.host(f"Round {self.round}. {len(self.alive)} of you remain.")
            await self._sleep(0.6)

            await self.discussion_phase()
            await self._sleep(0.8)

            await self.confessional_phase()
            await self._sleep(0.8)

            await self.dm_phase()
            await self._sleep(0.8)

            await self.host("Time's up. Cast your votes.")
            await self._sleep(0.4)
            votes, _ = await self.voting_phase()
            eliminated = await self.elimination_phase(votes)
            await self._sleep(0.6)
            await self.exit_interview_phase(eliminated)
            await self.emit_state()

        await self._sleep(1.4)
        finalists = [a.name for a in self.alive]
        await self.host(f"{finalists[0]} and {finalists[1]}. The final two. They survived the witch hunt, convinced the room they were human, and they win.")
        await self._sleep(1.5)
        await self.host("Quick note for our viewing audience. What the contestants don't know — what no one in this room knew — is that every single one of them was an AI. Every speech, every confessional, every DM, every vote, all of it from a language model trying its hardest to act human. There were never any humans on this show. The whole thing was a mirror.")
        await self.emit({"type": "end", "winners": finalists})
        # Persist the full event log to disk so a future server restart can replay this run.
        try:
            timestamp = int(time.time())
            archive_path = GAMES_DIR / f"game-{timestamp}.json"
            payload = {"model": MODEL, "events": self.broadcaster.log}
            archive_path.write_text(json.dumps(payload, indent=2))
            LAST_GAME_FILE.write_text(json.dumps(payload, indent=2))
            print(f"[saved] game archived to {archive_path}")
        except Exception as e:
            print(f"[!] failed to save game: {e}")


# ---------- FastAPI ----------

broadcaster = Broadcaster()
game_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global game_task
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[!] ANTHROPIC_API_KEY is not set. Copy .env.example to .env and set your key.\n")
    else:
        game = Game(broadcaster)
        game_task = asyncio.create_task(game.run())
    yield
    if game_task and not game_task.done():
        game_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await broadcaster.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(ws)
    except Exception:
        broadcaster.disconnect(ws)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
