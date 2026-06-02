"""Live neuroevolution: a competing population trained in the shared world.

A whole *population* of tiny numpy MLP brains lives in the world at once.
Every generation the best survive and the rest are culled and replaced by
**tournament-selected, uniform-crossover, mutated** offspring. The trainer:

* runs **batched ticks** under one lock per ~60Hz GUI release, so the sim can
  go far past the rendering rate (10,000+ steps/sec is normal);
* exposes everything the user might want to nudge as *live attributes* — the
  GUI panel can change population size, mutation strength, survivors, the
  number of steps per generation, the **training phase**, etc. while the loop
  is running;
* boosts mutation automatically when the best score plateaus, then anneals
  back when progress resumes (adaptive mutation, breaks local optima);
* keeps a *hall-of-fame* clone of the all-time best brain and re-injects it
  whenever progress stalls.

Training **phases** shape what the environment looks like and what counts as
good behaviour:

* ``move``   — single bright waypoint, no obstacles. Pure locomotion.
* ``forage`` — several food sources, mild obstacles. Smell-driven hunting.
* ``maze``   — food behind heavy obstacle walls. Pathfinding.
* ``hunt``   — predators chase fleeing prey (falls back to forage for non-
  predators).

It still mirrors :class:`Trainer`'s interface (``start`` / ``request_stop`` /
``drain`` / ``running`` emitting :class:`Progress`) so the training screen can
drive it the same way, and obeys :class:`SharedControl` for speed / pause /
user commands / manual reward.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .. import config
from ..sim.morphology import CreatureMorphology
from ..sim.world import World
from ..species import registry
from ..species.species import Species
from . import policy_io
from . import spaces as space_builder
from .live_control import Command, SharedControl
from .neuro import MLPPolicy
from .trainer import Progress

# ---------------------------------------------------------------------------
# Defaults (every one of these is also live-tunable through the GUI)
# ---------------------------------------------------------------------------
POP_SIZE = 16            # creatures alive at once
SURVIVORS = 4            # how many of the best carry over each generation
GEN_STEPS = 300          # world steps per generation before culling
MUTATION = 0.15          # std of weight noise when breeding a child
TOURNAMENT_K = 3         # how many to sample for tournament selection
TREAT_GAIN = 4.0         # how strongly Treat/Scold nudges the leader

# Difficulty ramp inside one phase.
BASE_TARGET_DIST = 150.0
DIST_PER_LEVEL = 70.0
OBSTACLES_PER_LEVEL = 1
MAX_OBSTACLES = 10
ADVANCE_FITNESS = 12.0
AUTO_WINS = 2

# Phase-specific environment + scoring tweaks.
PHASES = ("move", "forage", "maze", "hunt")
PHASE_CONFIG = {
    "move":   dict(food_count=1, food_smell=2.4, obstacle_mult=0.0,
                   description="Reach a single waypoint."),
    "forage": dict(food_count=4, food_smell=1.2, obstacle_mult=1.0,
                   description="Find and eat food sources."),
    "maze":   dict(food_count=3, food_smell=0.7, obstacle_mult=2.5,
                   description="Get to food past obstacles."),
    "hunt":   dict(food_count=0, food_smell=0.0, obstacle_mult=0.3,
                   description="Predators catch fleeing prey."),
}
# Auto-curriculum order (used when phase auto-advances).
PHASE_ORDER_FORAGER = ("move", "forage", "maze")
PHASE_ORDER_PREDATOR = ("move", "hunt", "maze")

# Predator-only.
PREY_COUNT = 3
PREY_FLEE_SPEED = 2.6


@dataclass
class _Individual:
    creature: object
    policy: MLPPolicy
    fitness: float = 0.0
    prev_dist: float = 0.0


class EvolutionTrainer:
    """A population trained against a phase-driven curriculum in the live world."""

    def __init__(self, world: World, ctrl: SharedControl,
                 morph: CreatureMorphology, role: str = "forager") -> None:
        self.world = world
        self.ctrl = ctrl
        self.morph = morph
        self.role = role
        self.obs_dim = space_builder.observation_dim(morph)
        self.act_dim = morph.num_actuators
        self._rng = np.random.default_rng()

        # Threading.
        self._thread: threading.Thread | None = None
        self._queue: "list[Progress]" = []
        self._qlock = threading.Lock()
        self._stop = threading.Event()
        self._running = False

        # Population state.
        self.population: list[_Individual] = []
        self.prey: list[object] = []
        self.generation = 0
        self.total_steps = 0
        self.best_fitness = float("-inf")
        self.best_policy: MLPPolicy | None = None

        # Same start, every generation.
        self.start_pos = self._centre()
        self.start_angle = 0.0

        # Live-tunables (the GUI writes these via commands).
        self.pop_size = POP_SIZE
        self.survivors = SURVIVORS
        self.gen_steps = GEN_STEPS
        self.user_mutation = MUTATION
        self.adaptive_mutation = True

        # Curriculum.
        self.phase = "move" if role != "predator" else "move"
        self.level = 1
        self.auto_advance = True

        # Internal bookkeeping.
        self._win_streak = 0
        self._gen_best = 0.0
        self._stagnation = 0

    # ----- control API ----------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def effective_mutation(self) -> float:
        """User-set rate plus an adaptive bonus that grows while we're stuck."""
        if not self.adaptive_mutation or self._stagnation < 3:
            return self.user_mutation
        bonus = min(0.45, 0.05 * (self._stagnation - 2))
        return float(min(0.9, self.user_mutation + bonus))

    def start(self, species: Species, total_timesteps: int = 0, env=None) -> None:
        if self._running:
            return
        self._species = species
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def drain(self) -> list[Progress]:
        with self._qlock:
            out = self._queue
            self._queue = []
        return out

    def _emit(self, **kw) -> None:
        with self._qlock:
            self._queue.append(Progress(**kw))

    # ----- world helpers --------------------------------------------------
    def _centre(self) -> tuple[float, float]:
        return self.world.width * 0.5, self.world.height * 0.5

    def _target_distance(self) -> float:
        d = BASE_TARGET_DIST + (self.level - 1) * DIST_PER_LEVEL
        return min(d, 0.42 * min(self.world.width, self.world.height))

    def _goal_point(self) -> tuple[float, float]:
        cx, cy = self.start_pos
        ang = self._rng.uniform(0, 2 * math.pi)
        d = self._target_distance() * self._rng.uniform(0.8, 1.15)
        x = float(np.clip(cx + math.cos(ang) * d, 50, self.world.width - 50))
        y = float(np.clip(cy + math.sin(ang) * d, 50, self.world.height - 50))
        return x, y

    def _phase_cfg(self) -> dict:
        return PHASE_CONFIG[self.phase]

    def _is_hunt_phase(self) -> bool:
        return self.phase == "hunt" and self.role == "predator"

    def _build_environment(self) -> None:
        """Rebuild a random environment for the current phase + level."""
        self.world.clear_food()
        self.world.clear_walls()
        # Goals.
        cfg = self._phase_cfg()
        if self._is_hunt_phase():
            for p in list(self.prey):
                self.world.remove_creature(p)
            self.prey = []
            for _ in range(PREY_COUNT):
                self.prey.append(self.world.add_creature(
                    CreatureMorphology(), self._goal_point(), role="forager"))
        else:
            # Predators in non-hunt phases still chase food markers.
            for _ in range(cfg["food_count"]):
                self.world.add_food(*self._goal_point(),
                                    smell_strength=cfg["food_smell"])
        # Obstacles scaled by phase + level.
        n_obs = int(min((self.level - 1) * OBSTACLES_PER_LEVEL
                        * cfg["obstacle_mult"], MAX_OBSTACLES))
        sx, sy = self.start_pos
        for _ in range(n_obs):
            for _try in range(8):
                x1 = float(self._rng.uniform(80, self.world.width - 80))
                y1 = float(self._rng.uniform(80, self.world.height - 80))
                if math.hypot(x1 - sx, y1 - sy) < 120:
                    continue
                ang = self._rng.uniform(0, 2 * math.pi)
                length = float(self._rng.uniform(70, 160))
                x2 = float(np.clip(x1 + math.cos(ang) * length, 40,
                                   self.world.width - 40))
                y2 = float(np.clip(y1 + math.sin(ang) * length, 40,
                                   self.world.height - 40))
                self.world.add_wall(x1, y1, x2, y2)
                break

    def _ensure_targets(self) -> None:
        cfg = self._phase_cfg()
        if self._is_hunt_phase():
            self.prey = [p for p in self.prey if p.alive]
            while len(self.prey) < PREY_COUNT:
                self.prey.append(self.world.add_creature(
                    CreatureMorphology(), self._goal_point(), role="forager"))
        else:
            live = [f for f in self.world.food if not f.eaten]
            self.world.food = live
            while len(self.world.food) < cfg["food_count"]:
                self.world.add_food(*self._goal_point(),
                                    smell_strength=cfg["food_smell"])

    def _target_dist(self, creature) -> float:
        cx, cy = creature.position
        if self._is_hunt_phase():
            tgt = self.world.nearest_creature(cx, cy, role="forager")
            pt = tgt.position if tgt else None
        else:
            f = self.world.nearest_food(cx, cy)
            pt = (f.x, f.y) if f else None
        if pt is None:
            return space_builder._world_diag(self.world)
        return math.hypot(pt[0] - cx, pt[1] - cy)

    def _drift_prey(self) -> None:
        for p in self.prey:
            if not p.alive:
                continue
            px, py = p.position
            pred = self.world.nearest_creature(px, py, role="predator")
            vx = self._rng.uniform(-1, 1)
            vy = self._rng.uniform(-1, 1)
            if pred is not None:
                dx, dy = px - pred.position[0], py - pred.position[1]
                d = math.hypot(dx, dy) + 1e-6
                vx += (dx / d) * 1.5
                vy += (dy / d) * 1.5
            n = math.hypot(vx, vy) + 1e-6
            p.hub.velocity = (vx / n * PREY_FLEE_SPEED * 30.0,
                              vy / n * PREY_FLEE_SPEED * 30.0)

    # ----- population ops -------------------------------------------------
    def _spawn_individual(self, policy: MLPPolicy) -> _Individual:
        creature = self.world.add_creature(
            self.morph, self.start_pos, self.start_angle, role=self.role)
        ind = _Individual(creature=creature, policy=policy)
        ind.prev_dist = self._target_dist(creature)
        return ind

    def _reset_positions(self) -> None:
        for ind in self.population:
            ind.creature.alive = True
            ind.creature.energy = config.START_ENERGY
            ind.creature.teleport(*self.start_pos)
            ind.fitness = 0.0
            ind.prev_dist = self._target_dist(ind.creature)

    def _tournament(self, pool: list[_Individual]) -> _Individual:
        k = min(TOURNAMENT_K, len(pool))
        if k <= 0:
            return self.population[0]
        idx = self._rng.choice(len(pool), size=k, replace=False)
        return max((pool[int(i)] for i in idx), key=lambda i: i.fitness)

    def _breed_child(self, pool: list[_Individual]) -> MLPPolicy:
        a = self._tournament(pool).policy
        b = self._tournament(pool).policy
        child = MLPPolicy.crossover(a, b, self._rng)
        return child.mutate(self.effective_mutation, self._rng)

    def _select_and_breed(self) -> None:
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        survivors = self.population[:max(1, self.survivors)]
        self._gen_best = survivors[0].fitness if survivors else 0.0
        improved = self._gen_best > self.best_fitness + 1e-3
        if improved:
            self.best_fitness = self._gen_best
            self.best_policy = survivors[0].policy.clone()
            self._stagnation = 0
        else:
            self._stagnation += 1

        # Cull losers.
        for loser in self.population[len(survivors):]:
            self.world.remove_creature(loser.creature)

        new_pop: list[_Individual] = []
        # Elite survivors carried over verbatim (their fitness resets).
        for s in survivors:
            s.fitness = 0.0
            s.prev_dist = self._target_dist(s.creature)
            new_pop.append(s)
        # Hall of fame: re-inject the all-time best when we're stuck.
        if (self.best_policy is not None and self._stagnation >= 3
                and len(new_pop) < self.pop_size):
            new_pop.append(self._spawn_individual(self.best_policy.clone()))
        # Fill the rest with crossover+mutation offspring.
        while len(new_pop) < self.pop_size:
            new_pop.append(self._spawn_individual(self._breed_child(survivors)))
        # If pop_size was shrunk live, trim the tail.
        while len(new_pop) > self.pop_size:
            extra = new_pop.pop()
            if extra.creature in self.world.creatures:
                self.world.remove_creature(extra.creature)
        self.population = new_pop

    # ----- curriculum -----------------------------------------------------
    def _advance_level(self) -> None:
        self.level += 1
        self._win_streak = 0
        self._build_environment()
        self._reset_positions()

    def _set_phase(self, name: str) -> None:
        if name not in PHASES or name == self.phase:
            return
        self.phase = name
        self.level = 1
        self._win_streak = 0
        self._stagnation = 0
        self._build_environment()
        self._reset_positions()

    def _next_phase(self) -> str | None:
        order = (PHASE_ORDER_PREDATOR if self.role == "predator"
                 else PHASE_ORDER_FORAGER)
        if self.phase not in order:
            return None
        idx = order.index(self.phase)
        return order[idx + 1] if idx + 1 < len(order) else None

    def _maybe_auto_advance(self) -> None:
        if not self.auto_advance:
            self._win_streak = 0
            return
        if self._gen_best >= ADVANCE_FITNESS:
            self._win_streak += 1
            if self._win_streak >= AUTO_WINS:
                # After a few levels in a phase, jump to the next phase.
                if self.level >= 4:
                    nxt = self._next_phase()
                    if nxt is not None:
                        self._set_phase(nxt)
                        return
                self._advance_level()
        else:
            self._win_streak = 0

    # ----- the loop -------------------------------------------------------
    def _run(self) -> None:
        try:
            with self.ctrl.lock:
                self._build_environment()
                for _ in range(self.pop_size):
                    self.population.append(self._spawn_individual(
                        MLPPolicy(self.obs_dim, self.act_dim, rng=self._rng)))
            self._emit(timesteps=0, total=0, mean_reward=0.0)

            step_in_gen = 0
            while not self._stop.is_set():
                # Pause check (no work while paused).
                paused, sps = self.ctrl.snapshot_speed()
                if paused:
                    time.sleep(0.05)
                    continue

                target_dt = (1.0 / sps) if sps > 0 else 0.0
                # Batch ticks so we release the lock only ~60 times/sec
                # regardless of the sim rate. Critical for 10k+ steps/sec.
                if sps > 120:
                    batch = max(1, min(int(sps / 60), 400))
                else:
                    batch = 1

                t0 = time.perf_counter()
                with self.ctrl.lock:
                    for _ in range(batch):
                        if self._stop.is_set():
                            break
                        self._tick()
                        step_in_gen += 1
                        self.total_steps += 1
                        if step_in_gen >= self.gen_steps:
                            self._select_and_breed()
                            self._build_environment()
                            self._reset_positions()
                            self._maybe_auto_advance()
                            step_in_gen = 0
                            self.generation += 1
                            self._emit(timesteps=self.total_steps, total=0,
                                       mean_reward=self.best_fitness)
                # Pace outside the lock so rendering can grab it.
                if target_dt > 0:
                    ideal = target_dt * batch
                    elapsed = time.perf_counter() - t0
                    if elapsed < ideal:
                        time.sleep(ideal - elapsed)

            self._finish_ok()
        except Exception as exc:  # surface to UI rather than hang
            self._emit(timesteps=0, total=0, mean_reward=0.0,
                       done=True, error=str(exc))
        finally:
            self._running = False

    def _tick(self) -> None:
        self._apply_commands()
        self._ensure_targets()

        # Every brain senses and acts.
        for ind in self.population:
            if not ind.creature.alive:
                continue
            obs = space_builder.observe(self.world, ind.creature)
            ind.creature.apply_action(ind.policy.act(obs))
        if self._is_hunt_phase():
            self._drift_prey()

        eats = self.world.step(config.SIM_DT)
        ate = {id(c) for c, _ in eats}
        caught = {id(p) for p, _ in self.world.catches}

        # Score: shaped progress + a big bonus for actually reaching the goal.
        for ind in self.population:
            c = ind.creature
            if not c.alive:
                continue
            dist = self._target_dist(c)
            ind.fitness += (ind.prev_dist - dist) * 0.05
            ind.prev_dist = dist
            if id(c) in ate or id(c) in caught:
                ind.fitness += 10.0
                ind.prev_dist = self._target_dist(c)

        # Caught prey: respawn elsewhere so the chase continues.
        if self.world.catches:
            for _pred, victim in self.world.catches:
                victim.alive = True
                victim.teleport(*self._goal_point())

        # Manual Treat / Scold lands on the current leader.
        bump = self.ctrl.take_reward()
        if bump and self.population:
            leader = max((i for i in self.population if i.creature.alive),
                         key=lambda i: i.fitness, default=None)
            if leader is not None:
                leader.fitness += bump * TREAT_GAIN

    def _apply_commands(self) -> None:
        leader = max((i for i in self.population if i.creature.alive),
                     key=lambda i: i.fitness, default=None)
        for cmd in self.ctrl.drain_commands():
            k = cmd.kind
            if k == "food":
                self.world.add_food(cmd.x, cmd.y,
                                    smell_strength=self._phase_cfg()["food_smell"])
            elif k == "clear_food":
                self.world.clear_food()
            elif k == "drag" and leader is not None:
                leader.creature.teleport(cmd.x, cmd.y)
            elif k == "reset_pose" and leader is not None:
                leader.creature.teleport(*self.start_pos)
            elif k == "wall":
                self.world.add_wall(cmd.x, cmd.y, cmd.x2, cmd.y2)
            elif k == "advance_level":
                self._advance_level()
            elif k == "auto_advance":
                self.auto_advance = bool(cmd.x)
            elif k == "set_mutation":
                self.user_mutation = float(np.clip(cmd.x, 0.0, 1.0))
            elif k == "set_pop":
                self.pop_size = int(np.clip(cmd.x, 2, 128))
                # Trim immediately if it shrank; growth happens at next gen.
                while len(self.population) > self.pop_size:
                    extra = self.population.pop()
                    if extra.creature in self.world.creatures:
                        self.world.remove_creature(extra.creature)
                while len(self.population) < self.pop_size:
                    self.population.append(self._spawn_individual(
                        MLPPolicy(self.obs_dim, self.act_dim, rng=self._rng)))
            elif k == "set_survivors":
                self.survivors = int(np.clip(cmd.x, 1, max(1, self.pop_size - 1)))
            elif k == "set_gen_steps":
                self.gen_steps = int(np.clip(cmd.x, 50, 5000))
            elif k == "set_phase":
                self._set_phase(cmd.label)

    # ----- finish ---------------------------------------------------------
    def _finish_ok(self) -> None:
        if self.best_policy is None and self.population:
            self.population.sort(key=lambda i: i.fitness, reverse=True)
            self.best_policy = self.population[0].policy.clone()
        if self.best_policy is not None:
            policy_io.save_policy(self._species.name, self.best_policy)
            sp = self._species
            sp.stats.timesteps += int(self.total_steps)
            sp.stats.mean_reward = float(
                self.best_fitness if self.best_fitness > float("-inf") else 0.0)
            sp.stats.last_trained = time.time()
            sp.obs_dim = self.obs_dim
            sp.act_dim = self.act_dim
            registry.save(sp)
        self._emit(timesteps=self.total_steps, total=0,
                   mean_reward=max(self.best_fitness, 0.0), done=True)
