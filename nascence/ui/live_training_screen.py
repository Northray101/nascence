"""Live, interactive training: watch a whole population evolve in real time.

The unified sandbox the project is built around. A *population* of creatures
lives and learns by neuroevolution inside a shared, persistent world rendered
here every frame. The right-hand panel gives you live, click-friendly control
over almost every training knob:

* **Phase** dropdown — *Move* (point-to-point), *Forage* (food), *Maze*
  (obstacles), or *Hunt* (predators chase prey).
* **Tools**: drop / clear food, drag the current leader, draw walls.
* **Treat / Scold**: shape the leader's behaviour right now.
* **Difficulty**: advance level manually or leave Auto on.
* **Tunables**: population, survivors, variation (mutation strength),
  generation length — all live, no restart.
* **Speed**: 2 → 10,000 ticks/sec. The trainer batches under one lock at high
  speeds, so the sim can sprint while the GUI keeps drawing at ~60Hz.

The best brain is saved automatically when you press *Save best & stop*.
"""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import (
    UIButton, UIDropDownMenu, UIHorizontalSlider, UILabel,
)

from .. import config
from ..render.camera import Camera
from ..render.sim_renderer import draw_world
from ..rl.evolution import (
    EvolutionTrainer, GEN_STEPS, MUTATION, POP_SIZE, SURVIVORS,
)
from ..rl.live_control import Command, SharedControl
from ..sim.world import World
from .screen import Screen

_PANEL_W = 340
_VIEW_W = config.WINDOW_WIDTH - _PANEL_W
_TREAT = 6.0
_LIVE_TIMESTEPS = 50_000_000  # effectively "until you stop"
_MAX_SPEED = 10_000

# Human-readable phase choices in the dropdown.
_PHASE_LABELS = {
    "move":   "Phase 1: Move (point-to-point)",
    "forage": "Phase 2: Forage (food + smell)",
    "maze":   "Phase 3: Maze (obstacles)",
    "hunt":   "Phase 4: Hunt (predator only)",
}


class LiveTrainingScreen(Screen):
    def __init__(self, app, species) -> None:
        super().__init__(app)
        self.species = species

        self.world = World()
        self.ctrl = SharedControl()
        self.ctrl.set_speed(120.0)
        self.trainer = EvolutionTrainer(
            self.world, self.ctrl,
            morph=species.morphology,
            role=getattr(species, "role", "forager"))

        self.camera = Camera(_VIEW_W, config.WINDOW_HEIGHT)
        self.camera.fit(self.world.width, self.world.height)
        self.viewport = pygame.Rect(0, 0, _VIEW_W, config.WINDOW_HEIGHT)

        self.tool = "food"
        self.paused = False
        self._wall_start: tuple[float, float] | None = None
        self._dragging = False
        self.reward_history: list[float] = []
        self.timesteps = 0
        self.status = "Spawning the first generation…"
        self.auto_advance = True

        self._build_panel()

    # -------- panel layout (a single, tightly-packed column) --------
    def _build_panel(self) -> None:
        px = _VIEW_W + 16
        w = _PANEL_W - 32
        m = self.manager

        UILabel(pygame.Rect((px, 6), (w, 22)),
                f"Training: {self.species.name}", m)

        # -- phase dropdown ------------------------------------------------
        UILabel(pygame.Rect((px, 30), (w, 20)), "Training phase:", m)
        phase_opts = list(_PHASE_LABELS.values())
        start_label = _PHASE_LABELS[self.trainer.phase]
        self.phase_menu = UIDropDownMenu(
            phase_opts, start_label,
            pygame.Rect((px, 50), (w, 28)), m)

        # -- tools row ------------------------------------------------------
        self.btn_food = UIButton(pygame.Rect((px, 84), (96, 28)), "Food", m)
        self.btn_drag = UIButton(pygame.Rect((px + 100, 84), (96, 28)), "Drag", m)
        self.btn_wall = UIButton(pygame.Rect((px + 200, 84), (108, 28)), "Wall", m)
        self.btn_clear = UIButton(
            pygame.Rect((px, 116), (w, 22)), "Clear all food", m)

        # -- teach now ------------------------------------------------------
        self.btn_treat = UIButton(pygame.Rect((px, 144), (148, 30)),
                                  "Treat  (+)", m)
        self.btn_scold = UIButton(pygame.Rect((px + 152, 144), (156, 30)),
                                  "Scold  (–)", m)

        # -- difficulty -----------------------------------------------------
        self.btn_level = UIButton(pygame.Rect((px, 180), (148, 28)),
                                  "Advance level ▶", m)
        self.btn_auto = UIButton(pygame.Rect((px + 152, 180), (156, 28)),
                                 "Auto: ON", m)

        # -- live tunables (label + slider per row) ------------------------
        self.pop_label = UILabel(pygame.Rect((px, 214), (w, 20)),
                                 f"Population: {POP_SIZE}", m)
        self.pop_slider = UIHorizontalSlider(
            pygame.Rect((px, 234), (w, 16)), start_value=POP_SIZE,
            value_range=(2, 48), manager=m)

        self.surv_label = UILabel(pygame.Rect((px, 252), (w, 20)),
                                  f"Survivors: {SURVIVORS}", m)
        self.surv_slider = UIHorizontalSlider(
            pygame.Rect((px, 272), (w, 16)), start_value=SURVIVORS,
            value_range=(1, 12), manager=m)

        self.var_label = UILabel(pygame.Rect((px, 290), (w, 20)),
                                 f"Variation: {MUTATION:.2f}", m)
        self.var_slider = UIHorizontalSlider(
            pygame.Rect((px, 310), (w, 16)), start_value=MUTATION * 100,
            value_range=(1, 80), manager=m)

        self.gen_label = UILabel(pygame.Rect((px, 328), (w, 20)),
                                 f"Gen length: {GEN_STEPS}", m)
        self.gen_slider = UIHorizontalSlider(
            pygame.Rect((px, 348), (w, 16)), start_value=GEN_STEPS,
            value_range=(80, 2000), manager=m)

        # -- speed (the marquee knob) --------------------------------------
        self.speed_label = UILabel(pygame.Rect((px, 368), (w, 20)),
                                   "Speed: 120 steps/s", m)
        self.speed_slider = UIHorizontalSlider(
            pygame.Rect((px, 388), (w, 20)), start_value=120,
            value_range=(2, _MAX_SPEED), manager=m)

        self.btn_pause = UIButton(pygame.Rect((px, 412), (w, 26)), "Pause", m)
        self.btn_save = UIButton(pygame.Rect((px, 442), (w, 30)),
                                 "Save best & stop", m)
        self.btn_back = UIButton(
            pygame.Rect((px, config.WINDOW_HEIGHT - 40), (w, 30)),
            "Back (save best)", m)

        self.chart_rect = pygame.Rect(px, 478, w, config.WINDOW_HEIGHT - 478 - 46)

    def on_enter(self) -> None:
        self.trainer.start(self.species, _LIVE_TIMESTEPS)

    # -------- events -------------------------------------------------------
    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            self._on_button(event.ui_element)
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            self._on_slider(event.ui_element)
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.phase_menu:
                self._on_phase_change(event.text)
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
            self.tool = "drag"; self.status = "Drag tool: click/drag the leader."
        elif el == self.btn_wall:
            self.tool = "wall"; self._wall_start = None
            self.status = "Wall tool: click start point, then end point."
        elif el == self.btn_clear:
            self.ctrl.push(Command("clear_food"))
        elif el == self.btn_treat:
            self.ctrl.add_reward(_TREAT); self.status = "Treat! (+reward)"
        elif el == self.btn_scold:
            self.ctrl.add_reward(-_TREAT); self.status = "Scold! (–reward)"
        elif el == self.btn_level:
            self.ctrl.push(Command("advance_level"))
            self.status = "Advancing to a harder level…"
        elif el == self.btn_auto:
            self.auto_advance = not self.auto_advance
            self.ctrl.push(Command("auto_advance",
                                   x=1.0 if self.auto_advance else 0.0))
            self.btn_auto.set_text("Auto: ON" if self.auto_advance else "Auto: OFF")
        elif el == self.btn_pause:
            self.paused = not self.paused
            self.ctrl.set_paused(self.paused)
            self.btn_pause.set_text("Resume" if self.paused else "Pause")
        elif el in (self.btn_save, self.btn_back):
            self.status = "Saving the best brain…"
            self.trainer.request_stop()
            self.ctrl.set_paused(False)

    def _on_slider(self, el) -> None:
        if el == self.speed_slider:
            sps = float(self.speed_slider.get_current_value())
            self.ctrl.set_speed(sps)
            self.speed_label.set_text(f"Speed: {int(sps)} steps/s")
        elif el == self.pop_slider:
            v = int(self.pop_slider.get_current_value())
            self.pop_label.set_text(f"Population: {v}")
            self.ctrl.push(Command("set_pop", x=float(v)))
        elif el == self.surv_slider:
            v = int(self.surv_slider.get_current_value())
            self.surv_label.set_text(f"Survivors: {v}")
            self.ctrl.push(Command("set_survivors", x=float(v)))
        elif el == self.var_slider:
            v = float(self.var_slider.get_current_value()) / 100.0
            self.var_label.set_text(f"Variation: {v:.2f}")
            self.ctrl.push(Command("set_mutation", x=v))
        elif el == self.gen_slider:
            v = int(self.gen_slider.get_current_value())
            self.gen_label.set_text(f"Gen length: {v}")
            self.ctrl.push(Command("set_gen_steps", x=float(v)))

    def _on_phase_change(self, text: str) -> None:
        for key, lbl in _PHASE_LABELS.items():
            if lbl == text:
                self.ctrl.push(Command("set_phase", label=key))
                self.reward_history.clear()
                self.status = f"Switching to {lbl}…"
                return

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

    # -------- per-frame ----------------------------------------------------
    def update(self, dt: float) -> None:
        super().update(dt)
        for p in self.trainer.drain():
            self.timesteps = p.timesteps
            if p.mean_reward != 0.0 or self.reward_history:
                self.reward_history.append(p.mean_reward)
            if not p.done and self.trainer.running:
                self.status = (
                    f"{self.trainer.phase.upper()}  "
                    f"Lv {self.trainer.level}  "
                    f"Gen {self.trainer.generation}  "
                    f"best {p.mean_reward:.1f}  "
                    f"μ={self.trainer.effective_mutation:.2f}"
                )
            if p.done:
                if p.error:
                    self.status = f"Stopped (error): {p.error}"
                else:
                    from .species_manager import SpeciesManager
                    self.app.set_screen(SpeciesManager(self.app))
                    return

    # -------- drawing ------------------------------------------------------
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
        if r.height < 40:
            return
        pygame.draw.rect(surface, (24, 30, 46), r, border_radius=8)
        pygame.draw.rect(surface, (70, 82, 110), r, width=2, border_radius=8)
        surface.blit(font.render("Best score / generation",
                                 True, (170, 185, 210)), (r.x + 8, r.y + 4))
        data = self.reward_history
        if len(data) < 2:
            return
        lo, hi = min(data), max(data)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        pad = 26
        plot = pygame.Rect(r.x + pad, r.y + pad, r.width - pad - 8,
                           r.height - pad - 8)
        n = len(data)
        pts = [(plot.x + plot.width * i / (n - 1),
                plot.bottom - plot.height * (v - lo) / (hi - lo))
               for i, v in enumerate(data)]
        pygame.draw.lines(surface, (120, 200, 255), False, pts, 2)
