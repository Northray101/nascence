"""Base class for full-window screens.

Each screen owns its own pygame_gui ``UIManager`` and builds its widgets in
``__init__``. Navigation creates a fresh screen instance, so there is no stale
state to clean up beyond disposing the manager on exit.
"""

from __future__ import annotations

import pygame
import pygame_gui

from .. import config


class Screen:
    def __init__(self, app) -> None:
        self.app = app
        self.size = (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.manager = pygame_gui.UIManager(self.size)
        self.bg_color = (18, 22, 34)

    # -- lifecycle ----------------------------------------------------------
    def on_enter(self) -> None:
        pass

    def on_exit(self) -> None:
        self.manager.clear_and_reset()

    # -- per-frame ----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        self.manager.process_events(event)
        self.on_event(event)

    def on_event(self, event: pygame.event.Event) -> None:
        """Override for screen-specific event handling."""

    def update(self, dt: float) -> None:
        self.manager.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(self.bg_color)
        self.pre_draw_ui(surface)
        self.manager.draw_ui(surface)
        self.post_draw_ui(surface)

    def pre_draw_ui(self, surface: pygame.Surface) -> None:
        """Draw beneath the GUI (e.g. the sandbox world)."""

    def post_draw_ui(self, surface: pygame.Surface) -> None:
        """Draw above the GUI (e.g. overlay text)."""
