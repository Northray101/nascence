"""Thread-safe channel between the GUI and a live training session.

The trainer runs PPO on a background thread while the GUI thread renders the
shared world and lets the user interfere. All cross-thread state lives here,
guarded by one lock:

* speed / pause          — how fast the learning loop runs (or halts)
* manual reward          — "treat" / "scold" buttons inject reward right now
* a command queue        — place/move food, drag the creature, add walls, etc.

The environment drains this each ``step``; the GUI only ever pushes into it.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class Command:
    kind: str          # "food" | "clear_food" | "drag" | "wall" | "reset_pose"
    x: float = 0.0
    y: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


class SharedControl:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._commands: deque[Command] = deque()
        self.paused = False
        # Target environment steps per second. A high value ~= "train fast".
        self.steps_per_sec = 30.0
        self._pending_reward = 0.0

    # -- GUI side (producer) ------------------------------------------------
    def push(self, cmd: Command) -> None:
        with self.lock:
            self._commands.append(cmd)

    def add_reward(self, amount: float) -> None:
        with self.lock:
            self._pending_reward += amount

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = paused

    def set_speed(self, steps_per_sec: float) -> None:
        with self.lock:
            self.steps_per_sec = max(0.0, steps_per_sec)

    # -- env side (consumer) ------------------------------------------------
    def drain_commands(self) -> list[Command]:
        with self.lock:
            out = list(self._commands)
            self._commands.clear()
            return out

    def take_reward(self) -> float:
        with self.lock:
            r = self._pending_reward
            self._pending_reward = 0.0
            return r

    def snapshot_speed(self) -> tuple[bool, float]:
        with self.lock:
            return self.paused, self.steps_per_sec
