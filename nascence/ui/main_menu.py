"""Main menu: the first thing the user sees."""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UILabel

from .. import config
from .screen import Screen


class MainMenu(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        cx = config.WINDOW_WIDTH // 2
        w, h, gap = 280, 56, 18
        top = 240

        UILabel(
            relative_rect=pygame.Rect((cx - 300, 110), (600, 60)),
            text="nascence",
            manager=self.manager,
            object_id="#title",
        )
        UILabel(
            relative_rect=pygame.Rect((cx - 300, 175), (600, 30)),
            text="train the brains of bacteria, then set them loose",
            manager=self.manager,
        )

        self.btn_species = UIButton(
            relative_rect=pygame.Rect((cx - w // 2, top), (w, h)),
            text="Species  (create & train)",
            manager=self.manager,
        )
        self.btn_sandbox = UIButton(
            relative_rect=pygame.Rect((cx - w // 2, top + (h + gap)), (w, h)),
            text="Sandbox  (spawn & watch)",
            manager=self.manager,
        )
        self.btn_quit = UIButton(
            relative_rect=pygame.Rect((cx - w // 2, top + 2 * (h + gap)), (w, h)),
            text="Quit",
            manager=self.manager,
        )

    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_species:
                from .species_manager import SpeciesManager

                self.app.set_screen(SpeciesManager(self.app))
            elif event.ui_element == self.btn_sandbox:
                from .sandbox_screen import SandboxScreen

                self.app.set_screen(SandboxScreen(self.app))
            elif event.ui_element == self.btn_quit:
                self.app.quit()
