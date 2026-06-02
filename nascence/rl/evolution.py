"""Neuroevolution in the live, shared world.

Instead of one brain slowly refined by PPO, this runs a *population* of brains
at once, all visible in the same sandbox:

* every individual is a creature in the shared world driven by its own
  :class:`MLPPolicy`;
* each step, every creature senses, acts, and the world advances once;
* fitness accrues from progress toward (and reaching) its goal — food for
  foragers, prey for predators — plus any manual Treat/Scold from the user;
* every *generation* (a fixed number of steps) the population is ranked: the
  best survive, the rest are **killed off** and replaced by mutated copies of
  the winners. Over generations the species gets visibly better.

It mirrors :class:`Trainer`'s interface (``start`` / ``request_stop`` /
``drain`` / ``running`` emitting :class:`Progress`) so the training screen
drives it the same way, and it obeys the same :class:`SharedControl` for
speed / pause / user commands / manual reward.
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

# -- tunables ---------------------------------------------------------------
POP_SIZE = 12          # creatures alive at once
SURVIVORS = 3          # how many of the best carry over each generation
GEN_STEPS = 400        # world steps per generation before culling
MUTATION = 0.18        # std of weight noise when breeding a child
SPAWN_RADIUS = 260.0   # how far new creatures appear from the world centre
FOOD_TARGET = 4        # foragers: keep at least this many food around
PREY_COUNT = 3         # predators: this many scripted prey to hunt
PREY_FLEE_SPEED = 2.6
TREAT_GAIN = 4.0       # how strongly a Treat/Scold nudges the leader's fitness


@dataclass
class _Individual:
    creature: object
    policy: MLPPolicy
    fitness: float = 0.0
    prev_dist: float = 0.0


class EvolutionTrainer:
    def __init__(self, world: World, ctrl: SharedControl,
                 morph: CreatureMorphology, role: str = "forager") -> None:
        self.world = world
        self.ctrl = ctrl
        self.morph = morph
        self.role = role
        self.obs_dim = space_builder.observation_dim(morph)
        self.act_dim = morph.num_actuators
        self._rng = np.random.default_rng()

        self._thread: threading.Thread | None = None
        self._queue: "list[Progress]" = []
        self._qlock = threading.Lock()
        self._stop = threading.Event()
        self._running = False

        self.population: list[_Individual] = []
        self.prey: list[object] = []
        self.generation = 0
        self.total_steps = 0
        self.best_fitness = float("-inf")
        self.best_policy: MLPPolicy | None = None

    @property
    def running(self) -> bool:
        return self._running

    # -- control ------------------------------------------------------------
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

    # -- world helpers ------------------------------------------------------
    def _centre(self) -> tuple[float, float]:
        return self.world.width * 0.5, self.world.height * 0.5

    def _random_point(self, radius: float) -> tuple[float, float]:
        cx, cy = self._centre()
        ang = self._rng.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(self._rng.uniform(0.05, 1.0))
        x = float(np.clip(cx + math.cos(ang) * r, 40, self.world.width - 40))
        y = float(np.clip(cy + math.sin(ang) * r, 40, self.world.height - 40))
        return x, y

    def _spawn_individual(self, policy: MLPPolicy) -> _Individual:
        x, y = self._random_point(SPAWN_RADIUS)
        ang = self._rng.uniform(0, 2 * math.pi)
        creature = self.world.add_creature(self.morph, (x, y), ang, role=self.role)
        ind = _Individual(creature=creature, policy=policy)
        ind.prev_dist = self._target_dist(creature)
        return ind

    def _target_dist(self, creature) -> float:
        cx, cy = creature.position
        if self.role == "predator":
            tgt = self.world.nearest_creature(cx, cy, role="forager")
            pt = tgt.position if tgt else None
        else:
            f = self.world.nearest_food(cx, cy)
            pt = (f.x, f.y) if f else None
        if pt is None:
            return space_builder._world_diag(self.world)
        return math.hypot(pt[0] - cx, pt[1] - cy)

    def _ensure_targets(self) -> None:
        """Keep something to chase: food for foragers, prey for predators."""
        if self.role == "predator":
            self.prey = [p for p in self.prey if p.alive]
            while len(self.prey) < PREY_COUNT:
                x, y = self._random_point(SPAWN_RADIUS * 1.4)
                self.prey.append(self.world.add_creature(
                    CreatureMorphology(), (x, y), role="forager"))
        else:
            live = [f for f in self.world.food if not f.eaten]
            self.world.food = live
            while len(self.world.food) < FOOD_TARGET:
                x, y = self._random_point(SPAWN_RADIUS * 1.3)
                self.world.add_food(x, y)

    def _drift_prey(self) -> None:
        """Scripted prey wander and flee the nearest predator a little."""
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

    # -- the loop -----------------------------------------------------------
    def _run(self) -> None:
        try:
            with self.ctrl.lock:
                self._ensure_targets()
                for _ in range(POP_SIZE):
                    self.population.append(self._spawn_individual(
                        MLPPolicy(self.obs_dim, self.act_dim, rng=self._rng)))
            self._emit(timesteps=0, total=0, mean_reward=0.0)

            step_in_gen = 0
            while not self._stop.is_set():
                self._await_clock()
                t0 = time.perf_counter()
                with self.ctrl.lock:
                    self._tick()
                step_in_gen += 1
                self.total_steps += 1

                if step_in_gen >= GEN_STEPS:
                    with self.ctrl.lock:
                        self._select_and_breed()
                    step_in_gen = 0
                    self.generation += 1
                    self._emit(timesteps=self.total_steps, total=0,
                               mean_reward=self.best_fitness)

                self._throttle(t0)

            self._finish_ok()
        except Exception as exc:  # surface to UI instead of hanging
            self._emit(timesteps=0, total=0, mean_reward=0.0,
                       done=True, error=str(exc))
        finally:
            self._running = False

    def _tick(self) -> None:
        # 1. user influence (place/clear food, drag the leader, walls).
        self._apply_commands()
        self._ensure_targets()

        # 2. every brain senses and acts.
        for ind in self.population:
            if not ind.creature.alive:
                continue
            obs = space_builder.observe(self.world, ind.creature)
            ind.creature.apply_action(ind.policy.act(obs))
        if self.role == "predator":
            self._drift_prey()

        # 3. advance the shared world once.
        eats = self.world.step(config.SIM_DT)
        ate = {id(c) for c, _ in eats}
        caught = {id(p) for p, _ in self.world.catches}

        # 4. score everyone on progress toward their goal (+ reaching it).
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

        # 5. respawn prey that got caught so the hunt continues.
        if self.world.catches:
            for _pred, victim in self.world.catches:
                victim.alive = True
                victim.teleport(*self._random_point(SPAWN_RADIUS * 1.4))

        # 6. manual Treat / Scold lands on the current leader.
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
            if cmd.kind == "food":
                self.world.add_food(cmd.x, cmd.y)
            elif cmd.kind == "clear_food":
                self.world.clear_food()
            elif cmd.kind == "drag" and leader is not None:
                leader.creature.teleport(cmd.x, cmd.y)
            elif cmd.kind == "reset_pose" and leader is not None:
                leader.creature.teleport(*self._centre())
            elif cmd.kind == "wall":
                self.world.add_wall(cmd.x, cmd.y, cmd.x2, cmd.y2)

    def _select_and_breed(self) -> None:
        """Rank the population; keep the best, cull and replace the rest."""
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        survivors = self.population[:SURVIVORS]
        if survivors and survivors[0].fitness > self.best_fitness:
            self.best_fitness = survivors[0].fitness
            self.best_policy = survivors[0].policy.clone()

        # Kill off the losers (they visibly disappear from the world).
        for loser in self.population[SURVIVORS:]:
            self.world.remove_creature(loser.creature)

        new_pop: list[_Individual] = []
        for s in survivors:
            s.fitness = 0.0
            s.prev_dist = self._target_dist(s.creature)
            new_pop.append(s)
        while len(new_pop) < POP_SIZE:
            parent = survivors[self._rng.integers(len(survivors))]
            child = parent.policy.mutate(MUTATION, self._rng)
            new_pop.append(self._spawn_individual(child))
        self.population = new_pop

    # -- pacing -------------------------------------------------------------
    def _await_clock(self) -> None:
        while not self._stop.is_set():
            paused, sps = self.ctrl.snapshot_speed()
            if not paused:
                self._target_dt = (1.0 / sps) if sps > 0 else 0.0
                return
            time.sleep(0.05)

    def _throttle(self, t0: float) -> None:
        dt = getattr(self, "_target_dt", 0.0)
        if dt > 0.0:
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # -- finish -------------------------------------------------------------
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
