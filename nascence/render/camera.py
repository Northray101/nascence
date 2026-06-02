"""World <-> screen coordinate transform with pan and zoom."""

from __future__ import annotations


class Camera:
    def __init__(self, viewport_w: int, viewport_h: int) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.zoom = 0.5
        # World point shown at the viewport centre.
        self.cx = 0.0
        self.cy = 0.0

    def center_on(self, x: float, y: float) -> None:
        self.cx = x
        self.cy = y

    def fit(self, world_w: float, world_h: float) -> None:
        """Zoom so the whole world fits, then centre on it."""
        zx = self.viewport_w / world_w
        zy = self.viewport_h / world_h
        self.zoom = min(zx, zy) * 0.95
        self.center_on(world_w * 0.5, world_h * 0.5)

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        sx = (x - self.cx) * self.zoom + self.viewport_w * 0.5
        sy = (y - self.cy) * self.zoom + self.viewport_h * 0.5
        return int(sx), int(sy)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        x = (sx - self.viewport_w * 0.5) / self.zoom + self.cx
        y = (sy - self.viewport_h * 0.5) / self.zoom + self.cy
        return x, y

    def pan(self, dx_screen: float, dy_screen: float) -> None:
        self.cx -= dx_screen / self.zoom
        self.cy -= dy_screen / self.zoom

    def zoom_by(self, factor: float) -> None:
        self.zoom = max(0.1, min(4.0, self.zoom * factor))
