from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    REVERSE = "REVERSE"
    NO_ACTION = "NO_ACTION"


class ActionSource(StrEnum):
    JEV = "JEV"
    RISK_ENGINE = "RISK_ENGINE"


EXECUTABLE_OPENS = frozenset({Action.OPEN_LONG, Action.OPEN_SHORT})
EXECUTABLE_EXITS = frozenset({Action.CLOSE, Action.REVERSE})
EXECUTABLE_ACTIONS = frozenset({*EXECUTABLE_OPENS, *EXECUTABLE_EXITS})
