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
        # Low, isotropic global damping; real thrust comes from leg drag.
        self.space.damping = config.FLUID_DAMPING

        self.creatures: list[Creature] = []
        self.food: list[Food] = []
        # User-placed walls, kept as endpoint pairs for rendering.
        self.user_walls: list[tuple[float, float, float, float]] = []
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

    def add_wall(self, x1: float, y1: float, x2: float, y2: float,
                 thickness: float = 8.0) -> None:
        seg = pymunk.Segment(self.space.static_body, (x1, y1), (x2, y2),
                             thickness)
        seg.friction = 0.5
        seg.collision_type = CT_WALL
        self.space.add(seg)
        self.user_walls.append((x1, y1, x2, y2))

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
    def _apply_swimming_drag(self) -> None:
        """Anisotropic drag on legs (paddle thrust) + light drag on each hub."""
        for creature in self.creatures:
            if not creature.alive:
                continue
            hub = creature.body.hub
            hv = hub.velocity
            hub.apply_force_at_world_point(
                (-config.BODY_DRAG * hv.x, -config.BODY_DRAG * hv.y),
                hub.position,
            )
            for leg in creature.body.legs:
                seg = leg.segment
                v = seg.velocity
                ca, sa = math.cos(seg.angle), math.sin(seg.angle)
                # Decompose velocity into along-leg (local x) and perpendicular.
                along = v.x * ca + v.y * sa
                perp = -v.x * sa + v.y * ca
                fa = -config.LEG_DRAG_ALONG * along
                fp = -config.LEG_DRAG_PERP * perp
                # Recompose into world space.
                fx = fa * ca - fp * sa
                fy = fa * sa + fp * ca
                seg.apply_force_at_world_point((fx, fy), seg.position)

    def step(self, dt: float = config.SIM_DT) -> list[tuple[Creature, Food]]:
        """Advance physics + smell one tick. Returns (creature, food) eats."""
        # Inject smell from every food source.
        for f in self.food:
            if not f.eaten:
                self.chem.emit(f.x, f.y, f.smell_strength)
        self.chem.step()

        # Step physics in substeps for stability, applying swimming drag each
        # substep (Pymunk clears applied forces after every step).
        sub_dt = dt / config.PHYSICS_SUBSTEPS
        for _ in range(config.PHYSICS_SUBSTEPS):
            self._apply_swimming_drag()
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
