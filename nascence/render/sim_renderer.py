"""Draw a :class:`World` onto a pygame surface via a :class:`Camera`."""

from __future__ import annotations

import numpy as np
import pygame

from ..sim.world import World
from .camera import Camera

# Colours
COL_BG = (12, 16, 28)
COL_WALL = (60, 70, 95)
COL_FOOD = (90, 220, 130)
COL_BODY = (120, 180, 255)
COL_BODY_EDGE = (200, 225, 255)
COL_LEG = (90, 140, 210)


def draw_world(
    surface: pygame.Surface,
    rect: pygame.Rect,
    world: World,
    camera: Camera,
    show_smell: bool = True,
) -> None:
    """Render ``world`` into ``rect`` on ``surface``."""
    prev_clip = surface.get_clip()
    surface.set_clip(rect)
    surface.fill(COL_BG, rect)

    if show_smell:
        _draw_smell(surface, rect, world, camera)

    _draw_walls(surface, world, camera)

    for (x1, y1, x2, y2) in world.user_walls:
        pygame.draw.line(surface, COL_WALL,
                         camera.world_to_screen(x1, y1),
                         camera.world_to_screen(x2, y2), 5)

    for f in world.food:
        if f.eaten:
            continue
        sx, sy = camera.world_to_screen(f.x, f.y)
        r = max(3, int(8 * camera.zoom))
        pygame.draw.circle(surface, COL_FOOD, (sx, sy), r)

    for creature in world.creatures:
        _draw_creature(surface, creature, camera)

    surface.set_clip(prev_clip)


def _draw_smell(surface, rect, world: World, camera: Camera) -> None:
    grid = world.chem.grid
    mx = float(grid.max())
    if mx <= 1e-6:
        return
    norm = np.clip(grid / mx, 0.0, 1.0)
    # Build a small RGBA surface and scale it to the world's screen extent.
    h, w = norm.shape
    rgb = np.zeros((w, h, 3), dtype=np.uint8)  # pygame wants (w, h, 3)
    intensity = (norm.T * 255).astype(np.uint8)
    rgb[..., 1] = (intensity * 0.45).astype(np.uint8)  # greenish glow
    rgb[..., 2] = (intensity * 0.30).astype(np.uint8)
    small = pygame.surfarray.make_surface(rgb)

    top_left = camera.world_to_screen(0, 0)
    bottom_right = camera.world_to_screen(world.width, world.height)
    dest_w = max(1, bottom_right[0] - top_left[0])
    dest_h = max(1, bottom_right[1] - top_left[1])
    scaled = pygame.transform.smoothscale(small, (dest_w, dest_h))
    surface.blit(scaled, top_left, special_flags=pygame.BLEND_ADD)


def _draw_walls(surface, world: World, camera: Camera) -> None:
    tl = camera.world_to_screen(0, 0)
    br = camera.world_to_screen(world.width, world.height)
    rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
    pygame.draw.rect(surface, COL_WALL, rect, width=2)


def _draw_creature(surface, creature, camera: Camera) -> None:
    # Legs first (so the body sits on top).
    for leg in creature.body.legs:
        verts = leg.shape.get_vertices()
        pts = []
        for v in verts:
            wx, wy = leg.segment.local_to_world(v)
            pts.append(camera.world_to_screen(wx, wy))
        if len(pts) >= 3:
            pygame.draw.polygon(surface, COL_LEG, pts)

    # Body hub.
    cx, cy = creature.position
    sx, sy = camera.world_to_screen(cx, cy)
    r = max(4, int(creature.morph.body_radius * camera.zoom))
    pygame.draw.circle(surface, COL_BODY, (sx, sy), r)
    pygame.draw.circle(surface, COL_BODY_EDGE, (sx, sy), r, width=2)

    # Heading indicator.
    import math

    hx = cx + math.cos(creature.angle) * creature.morph.body_radius
    hy = cy + math.sin(creature.angle) * creature.morph.body_radius
    pygame.draw.line(
        surface, COL_BODY_EDGE, (sx, sy), camera.world_to_screen(hx, hy), 2
    )
