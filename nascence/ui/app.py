"""The application controller: window, main loop, and screen switching."""

from __future__ import annotations

import pygame

from .. import config


class AppController:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.surface = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        self.running = True
        self.current = None

        # Imported here to avoid a circular import at module load.
        from .main_menu import MainMenu

        self.set_screen(MainMenu(self))

    def set_screen(self, screen) -> None:
        if self.current is not None:
            self.current.on_exit()
        self.current = screen
        self.current.on_enter()

    def quit(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(config.FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif self.current is not None:
                    self.current.handle_event(event)
            if self.current is not None:
                self.current.update(dt)
                self.current.draw(self.surface)
            pygame.display.flip()
        pygame.quit()
