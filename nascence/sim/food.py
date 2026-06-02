"""Food pellets that emit smell and can be eaten."""

from __future__ import annotations

from .. import config


class Food:
    """A simple food source.

    Food is intentionally not a physics body for the Phase-1 slice: it is a
    point in the world that emits smell and is "eaten" when a creature's body
    centre gets within :data:`config.FOOD_EAT_RADIUS`.
    """

    __slots__ = ("x", "y", "amount", "smell_strength", "eaten")

    def __init__(
        self,
        x: float,
        y: float,
        amount: float = config.ENERGY_PER_FOOD,
        smell_strength: float = config.FOOD_SMELL_STRENGTH,
    ) -> None:
        self.x = x
        self.y = y
        self.amount = amount
        self.smell_strength = smell_strength
        self.eaten = False

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)
