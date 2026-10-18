"""Rule variants. The defaults are the BarahAI standard ruleset (see README)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Rules:
    # A player who can capture must capture, and a bead that can keep jumping
    # must keep jumping. With False, both are optional and a STOP action ends
    # a chain early.
    forced_capture: bool = True
    # A capturing bead may jump again (changing direction freely) in the same turn.
    chain_capture: bool = True
    # Draw after this many consecutive turns without a capture (0 disables).
    quiet_limit: int = 40
    # Draw when the same position (with the same side to move) occurs this
    # many times (0 disables).
    repetition_limit: int = 3
    # Draw once this many turns have been played (0 disables). A safety net so
    # every episode terminates even with the other draw rules switched off.
    max_turns: int = 300
    # Which player makes the first move.
    first_player: int = 0

    def __post_init__(self) -> None:
        if self.first_player not in (0, 1):
            raise ValueError("first_player must be 0 or 1")
        for name in ("quiet_limit", "repetition_limit", "max_turns"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.repetition_limit == 1:
            raise ValueError("repetition_limit of 1 would draw every game at move one")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Rules:
        return cls(**data)


STANDARD = Rules()
