"""Threaded PPO training that keeps the GUI responsive.

``Trainer.start`` builds a PPO model on the species' environment and runs
``model.learn`` on a background thread. A callback streams progress
``(timesteps, mean_reward)`` into a thread-safe queue that the training screen
drains each frame to drive the progress bar and reward chart. When finished it
saves the brain and updates the species' training stats.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from ..species import registry
from ..species.species import Species
from . import policy_io


@dataclass
class Progress:
    timesteps: int
    total: int
    mean_reward: float
    done: bool = False
    error: str | None = None


class Trainer:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._queue: "queue.Queue[Progress]" = queue.Queue()
        self._stop = threading.Event()
        self._running = False
        self.last_mean_reward = 0.0
        self.last_timesteps = 0

    @property
    def running(self) -> bool:
        return self._running

    # -- control ------------------------------------------------------------
    def start(self, species: Species, total_timesteps: int) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, args=(species, total_timesteps), daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def drain(self) -> list[Progress]:
        """Return all progress updates queued since the last call."""
        out: list[Progress] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    # -- worker -------------------------------------------------------------
    def _run(self, species: Species, total_timesteps: int) -> None:
        try:
            # Heavy imports happen here, off the UI thread.
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback

            from .creature_env import CreatureEnv

            env = CreatureEnv(morph=species.morphology)
            model = PPO("MlpPolicy", env, device="cpu", verbose=0)

            trainer = self

            class _Cb(BaseCallback):
                def __init__(self) -> None:
                    super().__init__()
                    self._last_emit = 0.0

                def _on_step(self) -> bool:
                    if trainer._stop.is_set():
                        return False
                    now = time.time()
                    if now - self._last_emit >= 0.25:
                        self._last_emit = now
                        mean_r = trainer._recent_mean_reward(self.model)
                        trainer.last_mean_reward = mean_r
                        trainer.last_timesteps = self.num_timesteps
                        trainer._queue.put(
                            Progress(
                                timesteps=self.num_timesteps,
                                total=total_timesteps,
                                mean_reward=mean_r,
                            )
                        )
                    return True

            model.learn(total_timesteps=total_timesteps, callback=_Cb())

            policy_io.save_model(species.name, model)
            species.stats.timesteps += int(trainer.last_timesteps or total_timesteps)
            species.stats.mean_reward = float(trainer.last_mean_reward)
            species.stats.last_trained = time.time()
            species.obs_dim = int(env.observation_space.shape[0])
            species.act_dim = int(env.action_space.shape[0])
            registry.save(species)

            self._queue.put(
                Progress(
                    timesteps=trainer.last_timesteps or total_timesteps,
                    total=total_timesteps,
                    mean_reward=trainer.last_mean_reward,
                    done=True,
                )
            )
        except Exception as exc:  # surface errors to the UI rather than hang
            self._queue.put(
                Progress(timesteps=0, total=total_timesteps, mean_reward=0.0,
                         done=True, error=str(exc))
            )
        finally:
            self._running = False

    @staticmethod
    def _recent_mean_reward(model) -> float:
        buf = getattr(model, "ep_info_buffer", None)
        if buf and len(buf) > 0:
            rewards = [ep["r"] for ep in buf if "r" in ep]
            if rewards:
                return float(sum(rewards) / len(rewards))
        return 0.0
