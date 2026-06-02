"""Sandbox: spawn trained species into a live world, drop food, watch them go."""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIDropDownMenu, UILabel

from .. import config
from ..render.camera import Camera
from ..render.sim_renderer import draw_world
from ..rl import spaces as space_builder
from ..species import registry
from ..sim.world import World
from .screen import Screen

_PANEL_W = 320
_VIEW_W = config.WINDOW_WIDTH - _PANEL_W


class SandboxScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.world = World()
        self.camera = Camera(_VIEW_W, config.WINDOW_HEIGHT)
        self.camera.fit(self.world.width, self.world.height)
        self.viewport = pygame.Rect(0, 0, _VIEW_W, config.WINDOW_HEIGHT)

        self.playing = True
        self.tool = "food"  # "food" | "spawn"
        self.status = "Choose a species, press Spawn-mode, then click."

        # Loaded brains, shared per species; and which model drives each creature.
        self._models: dict[str, object] = {}
        self._creature_models: list[tuple[object, object]] = []

        px = _VIEW_W + 16
        UILabel(
            relative_rect=pygame.Rect((px, 16), (_PANEL_W - 32, 34)),
            text="Sandbox",
            manager=self.manager,
        )

        self._trained = [
            s.name for s in registry.list_species()
            if registry.has_trained_policy(s.name)
        ]
        options = self._trained or ["(no trained species yet)"]
        self.species_menu = UIDropDownMenu(
            options_list=options,
            starting_option=options[0],
            relative_rect=pygame.Rect((px, 60), (_PANEL_W - 32, 34)),
            manager=self.manager,
        )

        self.btn_spawn_mode = UIButton(
            relative_rect=pygame.Rect((px, 106), (140, 40)),
            text="Spawn mode",
            manager=self.manager,
        )
        self.btn_food_mode = UIButton(
            relative_rect=pygame.Rect((px + 150, 106), (140, 40)),
            text="Food mode",
            manager=self.manager,
        )
        self.btn_play = UIButton(
            relative_rect=pygame.Rect((px, 156), (140, 40)),
            text="Pause",
            manager=self.manager,
        )
        self.btn_reset = UIButton(
            relative_rect=pygame.Rect((px + 150, 156), (140, 40)),
            text="Reset",
            manager=self.manager,
        )
        self.btn_back = UIButton(
            relative_rect=pygame.Rect((px, config.WINDOW_HEIGHT - 60), (140, 40)),
            text="Back",
            manager=self.manager,
        )

    # -- events -------------------------------------------------------------
    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_spawn_mode:
                self.tool = "spawn"
                self.status = "Spawn mode: click in the world to place a bacterium."
            elif event.ui_element == self.btn_food_mode:
                self.tool = "food"
                self.status = "Food mode: click in the world to drop food."
            elif event.ui_element == self.btn_play:
                self.playing = not self.playing
                self.btn_play.set_text("Pause" if self.playing else "Play")
            elif event.ui_element == self.btn_reset:
                self._reset()
            elif event.ui_element == self.btn_back:
                from .main_menu import MainMenu

                self.app.set_screen(MainMenu(self.app))

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.viewport.collidepoint(event.pos):
                self._click_world(event.pos)
            elif event.button == 4:
                self.camera.zoom_by(1.1)
            elif event.button == 5:
                self.camera.zoom_by(1 / 1.1)

    def _selected_species(self) -> str | None:
        sel = self.species_menu.selected_option
        if isinstance(sel, (tuple, list)):
            sel = sel[0]
        if sel in self._trained:
            return sel
        return None

    def _click_world(self, pos) -> None:
        wx, wy = self.camera.screen_to_world(*pos)
        if self.tool == "food":
            self.world.add_food(wx, wy)
            return
        # Spawn mode.
        name = self._selected_species()
        if not name:
            self.status = "Train a species first (no trained brains yet)."
            return
        try:
            model = self._get_model(name)
        except Exception as exc:
            self.status = f"Could not load brain: {exc}"
            return
        species = registry.load(name)
        creature = self.world.add_creature(species.morphology, (wx, wy))
        self._creature_models.append((creature, model))
        self.status = f"Spawned a {name}."

    def _get_model(self, name: str):
        if name not in self._models:
            from ..rl import policy_io

            self._models[name] = policy_io.load_model(name)
        return self._models[name]

    def _reset(self) -> None:
        self.world = World()
        self.camera.fit(self.world.width, self.world.height)
        self._creature_models.clear()
        self.status = "Reset."

    # -- per-frame ----------------------------------------------------------
    def update(self, dt: float) -> None:
        super().update(dt)
        if not self.playing:
            return
        # Drive each creature's brain, then advance the shared world once.
        for creature, model in self._creature_models:
            if not creature.alive:
                continue
            obs = space_builder.observe(self.world, creature)
            action, _ = model.predict(obs, deterministic=True)
            creature.apply_action(action)
        self.world.step(config.SIM_DT)

    # -- drawing ------------------------------------------------------------
    def pre_draw_ui(self, surface: pygame.Surface) -> None:
        draw_world(surface, self.viewport, self.world, self.camera,
                   show_smell=True)

    def post_draw_ui(self, surface: pygame.Surface) -> None:
        font = pygame.font.SysFont("Helvetica", 16)
        surface.blit(
            font.render(self.status, True, (225, 235, 248)), (12, config.WINDOW_HEIGHT - 28)
        )
        mode = font.render(f"Tool: {self.tool}", True, (160, 220, 180))
        surface.blit(mode, (12, 10))
