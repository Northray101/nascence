"""A diffusing "smell" field over the world.

Food sources inject concentration into a coarse grid; each step the field
diffuses (blurs) and decays. Creatures sense the local concentration at a few
"nostril" points and the gradient (which way smell gets stronger).
"""

from __future__ import annotations

import numpy as np

from .. import config


class ChemicalField:
    def __init__(
        self,
        world_width: float = config.WORLD_WIDTH,
        world_height: float = config.WORLD_HEIGHT,
        cols: int = config.CHEM_GRID_COLS,
        rows: int = config.CHEM_GRID_ROWS,
    ) -> None:
        self.world_width = world_width
        self.world_height = world_height
        self.cols = cols
        self.rows = rows
        self.grid = np.zeros((rows, cols), dtype=np.float32)

    # -- coordinate helpers -------------------------------------------------
    def _cell(self, x: float, y: float) -> tuple[int, int]:
        c = int(np.clip(x / self.world_width * self.cols, 0, self.cols - 1))
        r = int(np.clip(y / self.world_height * self.rows, 0, self.rows - 1))
        return r, c

    # -- mutation -----------------------------------------------------------
    def emit(self, x: float, y: float, amount: float) -> None:
        """Add smell at a world position."""
        r, c = self._cell(x, y)
        self.grid[r, c] += amount

    def clear(self) -> None:
        self.grid.fill(0.0)

    def step(self) -> None:
        """Diffuse + decay one tick (cheap 4-neighbour blur)."""
        g = self.grid
        # Shifted neighbours (edges replicate by clamping via np.roll + fixups).
        up = np.roll(g, -1, axis=0)
        down = np.roll(g, 1, axis=0)
        left = np.roll(g, -1, axis=1)
        right = np.roll(g, 1, axis=1)
        neighbour_avg = (up + down + left + right) * 0.25
        d = config.CHEM_DIFFUSE
        self.grid = ((1.0 - d) * g + d * neighbour_avg) * config.CHEM_DECAY

    # -- sensing ------------------------------------------------------------
    def sample(self, x: float, y: float) -> float:
        r, c = self._cell(x, y)
        return float(self.grid[r, c])

    def gradient(self, x: float, y: float) -> tuple[float, float]:
        """Return a (gx, gy) unit-ish vector pointing toward stronger smell.

        Components are in world space, magnitude reflects the local slope.
        """
        r, c = self._cell(x, y)
        cw = self.world_width / self.cols
        ch = self.world_height / self.rows
        left = self.grid[r, max(c - 1, 0)]
        right = self.grid[r, min(c + 1, self.cols - 1)]
        up = self.grid[max(r - 1, 0), c]
        down = self.grid[min(r + 1, self.rows - 1), c]
        gx = (right - left) / (2.0 * cw)
        gy = (down - up) / (2.0 * ch)
        return float(gx), float(gy)
