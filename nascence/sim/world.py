"""The world: owns the Pymunk space, creatures, food and the smell field.

The same ``World`` is used headless during training and visualised in the
sandbox, so behaviour is identical in both.
"""

from __future__ import annotations

import math

import pymunk

from .. import config
from .chemical_field import ChemicalField
from .collision import CT_WALL
from .creature import Creature
from .food import Food
from .morphology import CreatureMorphology


class World:
    def __init__(
        self,
        width: float = config.WORLD_WIDTH,
        height: float = config.WORLD_HEIGHT,
    ) -> None:
        self.width = width
        self.height = height

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)  # top-down: no gravity
        # Velocity damping makes the world behave like viscous fluid so leg
        # strokes translate into net motion.
        self.space.damping = max(0.0, 1.0 - config.FLUID_DAMPING)

        self.creatures: list[Creature] = []
        self.food: list[Food] = []
        self.chem = ChemicalField(width, height)

        self._add_walls()

    # -- setup --------------------------------------------------------------
    def _add_walls(self) -> None:
        w, h, t = self.width, self.height, 10.0
        static = self.space.static_body
        segs = [
            pymunk.Segment(static, (0, 0), (w, 0), t),
            pymunk.Segment(static, (w, 0), (w, h), t),
            pymunk.Segment(static, (w, h), (0, h), t),
            pymunk.Segment(static, (0, h), (0, 0), t),
        ]
        for s in segs:
            s.friction = 0.5
            s.collision_type = CT_WALL
        self.space.add(*segs)

    # -- population ---------------------------------------------------------
    def add_creature(
        self,
        morph: CreatureMorphology,
        position: tuple[float, float],
        angle: float = 0.0,
    ) -> Creature:
        creature = Creature(morph, position, angle)
        self.space.add(creature.hub, creature.body.hub_shape)
        for leg in creature.body.legs:
            self.space.add(leg.segment, leg.shape)
        self.space.add(*creature.body.constraints)
        self.creatures.append(creature)
        return creature

    def remove_creature(self, creature: Creature) -> None:
        for c in creature.body.constraints:
            if c in self.space.constraints:
                self.space.remove(c)
        for s in creature.body.all_shapes():
            if s in self.space.shapes:
                self.space.remove(s)
        for b in creature.body.all_bodies():
            if b in self.space.bodies:
                self.space.remove(b)
        if creature in self.creatures:
            self.creatures.remove(creature)

    def add_food(self, x: float, y: float) -> Food:
        f = Food(x, y)
        self.food.append(f)
        return f

    def clear_food(self) -> None:
        self.food.clear()

    def nearest_food(self, x: float, y: float) -> Food | None:
        best: Food | None = None
        best_d2 = float("inf")
        for f in self.food:
            if f.eaten:
                continue
            d2 = (f.x - x) ** 2 + (f.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = f
        return best

    # -- simulation ---------------------------------------------------------
    def step(self, dt: float = config.SIM_DT) -> list[tuple[Creature, Food]]:
        """Advance physics + smell one tick. Returns (creature, food) eats."""
        # Inject smell from every food source.
        for f in self.food:
            if not f.eaten:
                self.chem.emit(f.x, f.y, f.smell_strength)
        self.chem.step()

        # Step physics in substeps for stability.
        sub_dt = dt / config.PHYSICS_SUBSTEPS
        for _ in range(config.PHYSICS_SUBSTEPS):
            self.space.step(sub_dt)

        # Handle eating (distance-based, see Food docstring).
        eats: list[tuple[Creature, Food]] = []
        for creature in self.creatures:
            if not creature.alive:
                continue
            cx, cy = creature.position
            for f in self.food:
                if f.eaten:
                    continue
                if math.hypot(f.x - cx, f.y - cy) <= config.FOOD_EAT_RADIUS:
                    f.eaten = True
                    creature.energy = min(1.0, creature.energy + f.amount)
                    eats.append((creature, f))
        return eats
