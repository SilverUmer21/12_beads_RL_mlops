"""Turn notation and game records.

A *turn* is one or more atomic actions by the same player. Notation follows
the board names from player 0's side: ``c2-c3`` for a step, ``b2xd4xb4`` for a
two-bead chain capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .board import square_name
from .rules import STANDARD, Rules
from .state import STOP, GameState, Move


@dataclass(slots=True)
class Turn:
    player: int
    moves: list[Move] = field(default_factory=list)

    @property
    def captures(self) -> int:
        return sum(m.is_capture for m in self.moves)

    def notation(self) -> str:
        moves = [m for m in self.moves if m.action != STOP]
        if not moves:
            return "--"
        sep = "x" if moves[0].is_capture else "-"
        return square_name(moves[0].src) + "".join(sep + square_name(m.dst) for m in moves)


@dataclass(slots=True)
class GameRecord:
    """Rules plus the action list: enough to replay a game exactly."""

    rules: Rules = STANDARD
    actions: list[int] = field(default_factory=list)

    def replay(self) -> list[GameState]:
        states = [GameState.new(self.rules)]
        for a in self.actions:
            states.append(states[-1].apply(a))
        return states

    def turns(self) -> list[Turn]:
        return group_turns(self.replay(), self.actions)

    def to_json(self) -> str:
        final = self.replay()[-1]
        result = None
        if final.result:
            result = {"winner": final.result.winner, "reason": final.result.reason}
        return json.dumps(
            {"format": "barahai-record/1", "rules": self.rules.to_dict(), "actions": self.actions, "result": result},
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> GameRecord:
        data = json.loads(text)
        if data.get("format") != "barahai-record/1":
            raise ValueError("not a BarahAI game record")
        record = cls(Rules.from_dict(data["rules"]), [int(a) for a in data["actions"]])
        record.replay()  # validates every action
        return record

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GameRecord:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def group_turns(states: list[GameState], actions: list[int]) -> list[Turn]:
    """Group atomic actions into turns. ``states[i]`` is the state before ``actions[i]``."""
    turns: list[Turn] = []
    for state, action in zip(states, actions):
        if not turns or state.chain < 0:
            turns.append(Turn(state.turn))
        turns[-1].moves.append(state.describe(action))
    return turns
