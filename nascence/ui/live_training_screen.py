"""Live, interactive training: watch a creature learn and meddle in real time.

This is the unified sandbox the project is built around. The creature trains
(PPO) inside a shared, persistent world rendered here every frame. While it
learns you can:

* drop / clear food (Food tool)
* drag the creature around (Drag tool)
* draw walls (Wall tool: click start, click end)
* reward it now (Treat) or punish it now (Scold)
* set the speed (slow to fast-forward) and pause

The brain is saved automatically when you press "Save & stop".
"""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIHorizontalSlider, UILabel

from .. import config
from ..render.camera import Camera
from ..render.sim_renderer import draw_world
from ..rl.live_control import Command, SharedControl
from ..rl.live_env import LiveCreatureEnv
from ..rl.trainer import Trainer
from ..sim.world import World
from .screen import Screen

_PANEL_W = 340
_VIEW_W = config.WINDOW_WIDTH - _PANEL_W
_TREAT = 6.0  # size of a manual reward / punishment
_LIVE_TIMESTEPS = 5_000_000  # effectively "until you stop"


class LiveTrainingScreen(Screen):
    def __init__(self, app, species) -> None:
        super().__init__(app)
        self.species = species

        # Shared world + control channel + the live env (all on this thread).
        self.world = World()
        self.ctrl = SharedControl()
        self.ctrl.set_speed(30.0)
        self.live_env = LiveCreatureEnv(self.world, self.ctrl,
                                        morph=species.morphology)
        self.trainer = Trainer()

        self.camera = Camera(_VIEW_W, config.WINDOW_HEIGHT)
        self.camera.fit(self.world.width, self.world.height)
        self.viewport = pygame.Rect(0, 0, _VIEW_W, config.WINDOW_HEIGHT)

        self.tool = "food"
        self.paused = False
        self._wall_start: tuple[float, float] | None = None
        self._dragging = False
        self.reward_history: list[float] = []
        self.timesteps = 0
        self.status = "Warming up the brain… (loading PyTorch)"

        self._build_panel()

    # -- panel --------------------------------------------------------------
    def _build_panel(self) -> None:
        px = _VIEW_W + 16
        w = _PANEL_W - 32
        UILabel(relative_rect=pygame.Rect((px, 12), (w, 30)),
                text=f"Training: {self.species.name}", manager=self.manager)

        UILabel(relative_rect=pygame.Rect((px, 50), (w, 24)),
                text="Tools (click in the world):", manager=self.manager)
        self.btn_food = UIButton(pygame.Rect((px, 78), (105, 38)),
                                 "Food", self.manager)
        self.btn_drag = UIButton(pygame.Rect((px + 110, 78), (95, 38)),
                                 "Drag", self.manager)
        self.btn_wall = UIButton(pygame.Rect((px + 210, 78), (95, 38)),
                                 "Wall", self.manager)
        self.btn_clear = UIButton(pygame.Rect((px, 120), (w, 30)),
                                  "Clear all food", self.manager)

        UILabel(relative_rect=pygame.Rect((px, 160), (w, 24)),
                text="Teach it right now:", manager=self.manager)
        self.btn_treat = UIButton(pygame.Rect((px, 188), (150, 44)),
                                  "Treat  (+)", self.manager)
        self.btn_scold = UIButton(pygame.Rect((px + 155, 188), (150, 44)),
                                  "Scold  (–)", self.manager)

        self.speed_label = UILabel(pygame.Rect((px, 248), (w, 24)),
                                   "Speed: 30 steps/s", self.manager)
        self.speed_slider = UIHorizontalSlider(
            pygame.Rect((px, 274), (w, 28)), start_value=30,
            value_range=(2, 400), manager=self.manager)
        self.btn_pause = UIButton(pygame.Rect((px, 308), (w, 38)),
                                  "Pause", self.manager)

        self.btn_save = UIButton(pygame.Rect((px, 360), (w, 44)),
                                 "Save & stop", self.manager)
        self.btn_back = UIButton(
            pygame.Rect((px, config.WINDOW_HEIGHT - 56), (w, 40)),
            "Back (save)", self.manager)

        self.chart_rect = pygame.Rect(px, 420, w, 200)

    def on_enter(self) -> None:
        self.trainer.start(self.species, _LIVE_TIMESTEPS, env=self.live_env)

    # -- events -------------------------------------------------------------
    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            self._on_button(event.ui_element)
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.speed_slider:
                sps = float(self.speed_slider.get_current_value())
                self.ctrl.set_speed(sps)
                self.speed_label.set_text(f"Speed: {int(sps)} steps/s")
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.viewport.collidepoint(event.pos):
                self._click_world(event.pos)
            elif event.button == 4:
                self.camera.zoom_by(1.1)
            elif event.button == 5:
                self.camera.zoom_by(1 / 1.1)
        elif event.type == pygame.MOUSEMOTION:
            if self._dragging and self.viewport.collidepoint(event.pos):
                wx, wy = self.camera.screen_to_world(*event.pos)
                self.ctrl.push(Command("drag", x=wx, y=wy))
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging = False

    def _on_button(self, el) -> None:
        if el == self.btn_food:
            self.tool = "food"; self.status = "Food tool: click to drop food."
        elif el == self.btn_drag:
            self.tool = "drag"; self.status = "Drag tool: click/drag the creature."
        elif el == self.btn_wall:
            self.tool = "wall"; self._wall_start = None
            self.status = "Wall tool: click start point, then end point."
        elif el == self.btn_clear:
            self.ctrl.push(Command("clear_food"))
        elif el == self.btn_treat:
            self.ctrl.add_reward(_TREAT); self.status = "Treat! (+reward)"
        elif el == self.btn_scold:
            self.ctrl.add_reward(-_TREAT); self.status = "Scold! (–reward)"
        elif el == self.btn_pause:
            self.paused = not self.paused
            self.ctrl.set_paused(self.paused)
            self.btn_pause.set_text("Resume" if self.paused else "Pause")
        elif el in (self.btn_save, self.btn_back):
            self.status = "Saving brain…"
            self.trainer.request_stop()
            self.ctrl.set_paused(False)  # let the loop reach the stop check

    def _click_world(self, pos) -> None:
        wx, wy = self.camera.screen_to_world(*pos)
        if self.tool == "food":
            self.ctrl.push(Command("food", x=wx, y=wy))
        elif self.tool == "drag":
            self._dragging = True
            self.ctrl.push(Command("drag", x=wx, y=wy))
        elif self.tool == "wall":
            if self._wall_start is None:
                self._wall_start = (wx, wy)
            else:
                x1, y1 = self._wall_start
                self.ctrl.push(Command("wall", x=x1, y=y1, x2=wx, y2=wy))
                self._wall_start = None

    # -- per-frame ----------------------------------------------------------
    def update(self, dt: float) -> None:
        super().update(dt)
        for p in self.trainer.drain():
            self.timesteps = p.timesteps
            if p.mean_reward != 0.0 or self.reward_history:
                self.reward_history.append(p.mean_reward)
            if not p.done and self.trainer.running:
                self.status = (
                    f"Learning…  {p.timesteps:,} steps   "
                    f"avg reward: {p.mean_reward:.1f}"
                )
            if p.done:
                if p.error:
                    self.status = f"Stopped (error): {p.error}"
                else:
                    from .species_manager import SpeciesManager
                    self.app.set_screen(SpeciesManager(self.app))
                    return

    # -- drawing ------------------------------------------------------------
    def pre_draw_ui(self, surface: pygame.Surface) -> None:
        with self.ctrl.lock:
            draw_world(surface, self.viewport, self.world, self.camera,
                       show_smell=True)
        if self._wall_start is not None:
            sx, sy = self.camera.world_to_screen(*self._wall_start)
            pygame.draw.circle(surface, (240, 200, 120), (sx, sy), 6)

    def post_draw_ui(self, surface: pygame.Surface) -> None:
        font = pygame.font.SysFont("Helvetica", 16)
        surface.blit(font.render(f"Tool: {self.tool}", True, (160, 220, 180)),
                     (12, 10))
        surface.blit(font.render(self.status, True, (225, 235, 248)),
                     (12, config.WINDOW_HEIGHT - 26))
        self._draw_chart(surface, font)

    def _draw_chart(self, surface, font) -> None:
        r = self.chart_rect
        pygame.draw.rect(surface, (24, 30, 46), r, border_radius=8)
        pygame.draw.rect(surface, (70, 82, 110), r, width=2, border_radius=8)
        surface.blit(font.render("Avg reward (it's learning if this rises)",
                                 True, (170, 185, 210)), (r.x + 8, r.y + 6))
        data = self.reward_history
        if len(data) < 2:
            return
        lo, hi = min(data), max(data)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        pad = 34
        plot = pygame.Rect(r.x + pad, r.y + pad, r.width - pad - 12,
                           r.height - pad - 16)
        n = len(data)
        pts = [(plot.x + plot.width * i / (n - 1),
                plot.bottom - plot.height * (v - lo) / (hi - lo))
               for i, v in enumerate(data)]
        pygame.draw.lines(surface, (120, 200, 255), False, pts, 2)
