"""Training screen: pick how long to train, watch the reward climb, save."""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIDropDownMenu, UILabel

from .. import config
from ..rl.trainer import Trainer
from ..species.species import Species
from .screen import Screen


class TrainingScreen(Screen):
    def __init__(self, app, species: Species) -> None:
        super().__init__(app)
        self.species = species
        self.trainer = Trainer()
        self.progress = 0.0
        self.reward_history: list[float] = []
        self.status = "Pick a length and press Start."

        UILabel(
            relative_rect=pygame.Rect((30, 20), (700, 40)),
            text=f"Training: {species.name}",
            manager=self.manager,
        )

        UILabel(
            relative_rect=pygame.Rect((30, 80), (160, 30)),
            text="Training length:",
            manager=self.manager,
        )
        presets = list(config.TRAIN_PRESETS.keys())
        self.preset_menu = UIDropDownMenu(
            options_list=presets,
            starting_option=config.DEFAULT_PRESET,
            relative_rect=pygame.Rect((190, 78), (180, 34)),
            manager=self.manager,
        )

        self.btn_start = UIButton(
            relative_rect=pygame.Rect((400, 76), (140, 40)),
            text="Start",
            manager=self.manager,
        )
        self.btn_stop = UIButton(
            relative_rect=pygame.Rect((550, 76), (140, 40)),
            text="Stop & save",
            manager=self.manager,
        )
        self.btn_stop.disable()

        self.btn_back = UIButton(
            relative_rect=pygame.Rect((30, config.WINDOW_HEIGHT - 70), (160, 44)),
            text="Back",
            manager=self.manager,
        )

        # Geometry for the hand-drawn progress bar + reward chart.
        self.bar_rect = pygame.Rect(30, 150, config.WINDOW_WIDTH - 60, 28)
        self.chart_rect = pygame.Rect(30, 210, config.WINDOW_WIDTH - 60, 380)

    # -- events -------------------------------------------------------------
    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_start:
                self._start()
            elif event.ui_element == self.btn_stop:
                self.trainer.request_stop()
                self.status = "Stopping… (saving brain)"
            elif event.ui_element == self.btn_back:
                if self.trainer.running:
                    self.trainer.request_stop()
                from .species_manager import SpeciesManager

                self.app.set_screen(SpeciesManager(self.app))

    def _start(self) -> None:
        if self.trainer.running:
            return
        label = self.preset_menu.selected_option
        if isinstance(label, (tuple, list)):  # some versions return (text, id)
            label = label[0]
        total = config.TRAIN_PRESETS.get(label, config.TRAIN_PRESETS["Quick"])
        self.reward_history.clear()
        self.progress = 0.0
        self.status = "Warming up the brain… (first start loads PyTorch)"
        self.trainer.start(self.species, total)
        self.btn_start.disable()
        self.btn_stop.enable()

    # -- per-frame ----------------------------------------------------------
    def update(self, dt: float) -> None:
        super().update(dt)
        for p in self.trainer.drain():
            if p.total > 0:
                self.progress = min(1.0, p.timesteps / p.total)
            if p.mean_reward != 0.0 or self.reward_history:
                self.reward_history.append(p.mean_reward)
            if p.done:
                self.btn_start.enable()
                self.btn_stop.disable()
                if p.error:
                    self.status = f"Training failed: {p.error}"
                else:
                    self.progress = 1.0
                    self.status = (
                        f"Saved!  Avg reward: {p.mean_reward:.1f}. "
                        "Go to the Sandbox to spawn it."
                    )
            elif self.trainer.running:
                self.status = (
                    f"Training…  {int(self.progress * 100)}%   "
                    f"avg reward: {p.mean_reward:.1f}"
                )

    # -- drawing ------------------------------------------------------------
    def post_draw_ui(self, surface: pygame.Surface) -> None:
        font = pygame.font.SysFont("Helvetica", 18)
        surface.blit(
            font.render(self.status, True, (220, 230, 245)), (30, 122)
        )
        self._draw_progress(surface)
        self._draw_chart(surface, font)

    def _draw_progress(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, (40, 48, 66), self.bar_rect, border_radius=6)
        fill = self.bar_rect.copy()
        fill.width = int(self.bar_rect.width * self.progress)
        if fill.width > 0:
            pygame.draw.rect(surface, (90, 200, 130), fill, border_radius=6)
        pygame.draw.rect(surface, (90, 100, 130), self.bar_rect, width=2,
                         border_radius=6)

    def _draw_chart(self, surface: pygame.Surface, font) -> None:
        r = self.chart_rect
        pygame.draw.rect(surface, (24, 30, 46), r, border_radius=8)
        pygame.draw.rect(surface, (70, 82, 110), r, width=2, border_radius=8)
        label = font.render("Average reward over time (higher = learning)",
                            True, (170, 185, 210))
        surface.blit(label, (r.x + 12, r.y + 8))

        data = self.reward_history
        if len(data) < 2:
            return
        lo, hi = min(data), max(data)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        pad = 40
        plot = pygame.Rect(r.x + pad, r.y + pad, r.width - 2 * pad,
                           r.height - 2 * pad)
        pts = []
        n = len(data)
        for i, v in enumerate(data):
            x = plot.x + (plot.width * i) / (n - 1)
            y = plot.bottom - plot.height * (v - lo) / (hi - lo)
            pts.append((x, y))
        pygame.draw.lines(surface, (120, 200, 255), False, pts, 2)
        surface.blit(font.render(f"{hi:.1f}", True, (150, 165, 190)),
                     (r.x + 6, plot.y - 4))
        surface.blit(font.render(f"{lo:.1f}", True, (150, 165, 190)),
                     (r.x + 6, plot.bottom - 12))
