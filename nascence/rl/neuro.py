"""A tiny numpy neural-network brain for neuroevolution.

Unlike PPO (one brain refined by gradient descent), evolution runs a *whole
population* of these little nets at once: the best are kept, the rest are culled
and replaced by mutated copies of the winners. A plain numpy MLP is perfect for
this — it builds instantly, runs dozens in parallel with no torch, and mutates
by simply jittering its weights.

The policy exposes ``predict(obs, deterministic=True) -> (action, None)`` so a
saved brain drops straight into the Sandbox in place of an SB3 model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MLPPolicy:
    """A two-layer tanh MLP: obs -> hidden -> action, all in [-1, 1]."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 24,
        weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)
        if weights is None:
            rng = rng or np.random.default_rng()
            self.w1 = rng.normal(0.0, 0.5, (self.obs_dim, self.hidden)).astype(np.float32)
            self.b1 = np.zeros(self.hidden, dtype=np.float32)
            self.w2 = rng.normal(0.0, 0.5, (self.hidden, self.act_dim)).astype(np.float32)
            self.b2 = np.zeros(self.act_dim, dtype=np.float32)
        else:
            self.w1, self.b1, self.w2, self.b2 = weights

    # -- inference ----------------------------------------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)

    def predict(self, obs, deterministic: bool = True):
        """SB3-compatible signature so the Sandbox can drive evolved brains."""
        return self.act(obs), None

    # -- evolution ----------------------------------------------------------
    def clone(self) -> "MLPPolicy":
        return MLPPolicy(
            self.obs_dim, self.act_dim, self.hidden,
            (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()),
        )

    def mutate(self, rate: float, rng: np.random.Generator) -> "MLPPolicy":
        """Return a child with Gaussian noise (std=``rate``) added to weights."""
        def jit(a: np.ndarray) -> np.ndarray:
            return (a + rng.normal(0.0, rate, a.shape)).astype(np.float32)

        return MLPPolicy(
            self.obs_dim, self.act_dim, self.hidden,
            (jit(self.w1), jit(self.b1), jit(self.w2), jit(self.b2)),
        )

    @staticmethod
    def crossover(a: "MLPPolicy", b: "MLPPolicy",
                  rng: np.random.Generator) -> "MLPPolicy":
        """Uniform crossover: each weight randomly picked from ``a`` or ``b``.

        Combines what made two winners successful, instead of refining a single
        parent — escapes local optima much faster than pure mutation."""
        def cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            mask = rng.random(x.shape) < 0.5
            return np.where(mask, x, y).astype(np.float32)

        return MLPPolicy(
            a.obs_dim, a.act_dim, a.hidden,
            (cross(a.w1, b.w1), cross(a.b1, b.b1),
             cross(a.w2, b.w2), cross(a.b2, b.b2)),
        )

    # -- persistence --------------------------------------------------------
    def save_npz(self, path: Path | str) -> None:
        np.savez(
            str(path), w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            hidden=np.array(self.hidden),
        )

    @classmethod
    def load_npz(cls, path: Path | str) -> "MLPPolicy":
        d = np.load(str(path))
        w1, b1, w2, b2 = d["w1"], d["b1"], d["w2"], d["b2"]
        return cls(w1.shape[0], w2.shape[1], int(d["hidden"]), (w1, b1, w2, b2))
