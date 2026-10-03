from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    """One in-memory example.

    ``uid`` is safe to persist. ``text`` and ``reference`` are confidential and
    must stay in memory unless private error output is explicitly enabled.
    """

    uid: str
    text: str
    label: str
    reference: str | None
    source: str
    source_index: int


@dataclass(frozen=True)
class Prediction:
    uid: str
    label: str
    rationale: str
    parse_error: bool = False

