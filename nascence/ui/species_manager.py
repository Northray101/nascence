"""Species manager: create, list, train and delete named species."""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import (
    UIButton,
    UIDropDownMenu,
    UIHorizontalSlider,
    UILabel,
    UISelectionList,
    UITextEntryLine,
)
from pygame_gui.elements.ui_window import UIWindow

from .. import config
from ..sim.morphology import CreatureMorphology
from ..species import registry
from .screen import Screen


class _NewSpeciesDialog(UIWindow):
    """Small modal to name a species and pick its leg count."""

    def __init__(self, manager) -> None:
        super().__init__(
            rect=pygame.Rect((0, 0), (380, 470)),
            manager=manager,
            window_display_title="New species",
            object_id="#new_species",
        )
        self.set_blocking(True)
        w = 340

        UILabel(pygame.Rect((10, 8), (w, 24)), "Name your bacteria:",
                manager, container=self)
        self.name_entry = UITextEntryLine(pygame.Rect((10, 34), (w, 34)),
                                          manager, container=self)
        self.name_entry.set_text("Wiggler")

        UILabel(pygame.Rect((10, 76), (w, 22)), "Role:", manager, container=self)
        self.role_menu = UIDropDownMenu(
            ["forager (eats food)", "predator (hunts prey)"],
            "forager (eats food)", pygame.Rect((10, 100), (w, 32)),
            manager, container=self)

        UILabel(pygame.Rect((10, 140), (w, 22)), "Body:", manager, container=self)
        self.body_menu = UIDropDownMenu(
            ["rigid (firm)", "jelly (squishy)"], "rigid (firm)",
            pygame.Rect((10, 164), (w, 32)), manager, container=self)

        self.legs_label = UILabel(pygame.Rect((10, 204), (w, 24)), "Legs: 3",
                                  manager, container=self)
        self.legs_slider = UIHorizontalSlider(
            pygame.Rect((10, 230), (w, 26)), start_value=3, value_range=(2, 6),
            manager=manager, container=self)

        self.seg_label = UILabel(pygame.Rect((10, 264), (w, 24)),
                                 "Leg joints: 1 (simple)", manager, container=self)
        self.seg_slider = UIHorizontalSlider(
            pygame.Rect((10, 290), (w, 26)), start_value=1, value_range=(1, 2),
            manager=manager, container=self)

        self.firm_label = UILabel(pygame.Rect((10, 324), (w, 24)),
                                  "Jelly wobble: firm", manager, container=self)
        self.firm_slider = UIHorizontalSlider(
            pygame.Rect((10, 350), (w, 26)), start_value=100, value_range=(20, 100),
            manager=manager, container=self)

        self.create_btn = UIButton(pygame.Rect((10, 390), (160, 38)), "Create",
                                   manager, container=self)
        self.cancel_btn = UIButton(pygame.Rect((190, 390), (160, 38)), "Cancel",
                                   manager, container=self)

    @property
    def leg_count(self) -> int:
        return int(round(self.legs_slider.get_current_value()))

    @property
    def leg_segments(self) -> int:
        return int(round(self.seg_slider.get_current_value()))

    @property
    def role(self) -> str:
        return "predator" if "predator" in str(self.role_menu.selected_option) else "forager"

    @property
    def body_type(self) -> str:
        return "jelly" if "jelly" in str(self.body_menu.selected_option) else "rigid"

    @property
    def firmness(self) -> float:
        return float(self.firm_slider.get_current_value()) / 100.0


class SpeciesManager(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.dialog: _NewSpeciesDialog | None = None

        UILabel(
            relative_rect=pygame.Rect((30, 20), (400, 40)),
            text="Species",
            manager=self.manager,
        )

        self.listing = UISelectionList(
            relative_rect=pygame.Rect((30, 80), (560, 560)),
            item_list=[],
            manager=self.manager,
        )

        bx = 620
        self.btn_new = UIButton(
            relative_rect=pygame.Rect((bx, 80), (200, 48)),
            text="New species",
            manager=self.manager,
        )
        self.btn_train = UIButton(
            relative_rect=pygame.Rect((bx, 140), (200, 48)),
            text="Train selected",
            manager=self.manager,
        )
        self.btn_delete = UIButton(
            relative_rect=pygame.Rect((bx, 200), (200, 48)),
            text="Delete selected",
            manager=self.manager,
        )
        self.btn_back = UIButton(
            relative_rect=pygame.Rect((bx, 620), (200, 48)),
            text="Back",
            manager=self.manager,
        )
        self.hint = UILabel(
            relative_rect=pygame.Rect((bx, 280), (220, 200)),
            text="",
            manager=self.manager,
        )

        self._names: list[str] = []
        self._refresh()

    # -- data ---------------------------------------------------------------
    def _refresh(self) -> None:
        species = registry.list_species()
        self._names = [s.name for s in species]
        items = []
        for s in species:
            mark = "✓ trained" if s.trained else "· untrained"
            items.append(f"{s.name}   ({mark})")
        self.listing.set_item_list(items)

    def _selected_name(self) -> str | None:
        sel = self.listing.get_single_selection()
        if not sel:
            return None
        # Display text is "name   (trained/untrained)"; map it back to a name.
        for name in self._names:
            if sel.startswith(name + " "):
                return name
        return None

    # -- events -------------------------------------------------------------
    def on_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.btn_back:
                from .main_menu import MainMenu

                self.app.set_screen(MainMenu(self.app))
            elif event.ui_element == self.btn_new:
                if self.dialog is None:
                    self.dialog = _NewSpeciesDialog(self.manager)
            elif event.ui_element == self.btn_train:
                self._train_selected()
            elif event.ui_element == self.btn_delete:
                self._delete_selected()
            elif self.dialog is not None and event.ui_element == self.dialog.create_btn:
                self._create_from_dialog()
            elif self.dialog is not None and event.ui_element == self.dialog.cancel_btn:
                self._close_dialog()

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            d = self.dialog
            if d is None:
                return
            if event.ui_element == d.legs_slider:
                d.legs_label.set_text(f"Legs: {d.leg_count}")
            elif event.ui_element == d.seg_slider:
                kind = "1 (simple)" if d.leg_segments == 1 else "2 (hip + knee)"
                d.seg_label.set_text(f"Leg joints: {kind}")
            elif event.ui_element == d.firm_slider:
                f = d.firmness
                word = "firm" if f > 0.7 else ("wobbly" if f > 0.4 else "very wobbly")
                d.firm_label.set_text(f"Jelly wobble: {word}")

        elif event.type == pygame_gui.UI_WINDOW_CLOSE:
            if self.dialog is not None and event.ui_element == self.dialog:
                self.dialog = None

    # -- actions ------------------------------------------------------------
    def _create_from_dialog(self) -> None:
        assert self.dialog is not None
        d = self.dialog
        name = d.name_entry.get_text().strip() or "Wiggler"
        if registry.exists(name):
            self.hint.set_text("Name already exists.")
            return
        morph = CreatureMorphology(
            num_legs=d.leg_count,
            leg_segments=d.leg_segments,
            body_type=d.body_type,
            firmness=d.firmness,
        )
        registry.create(name, morph, role=d.role)
        self._close_dialog()
        self._refresh()
        self.hint.set_text(f"Created '{name}'.\nSelect it and Train.")

    def _close_dialog(self) -> None:
        if self.dialog is not None:
            self.dialog.kill()
            self.dialog = None

    def _train_selected(self) -> None:
        name = self._selected_name()
        if not name:
            self.hint.set_text("Select a species first.")
            return
        species = registry.load(name)
        from .live_training_screen import LiveTrainingScreen

        self.app.set_screen(LiveTrainingScreen(self.app, species))

    def _delete_selected(self) -> None:
        name = self._selected_name()
        if not name:
            self.hint.set_text("Select a species first.")
            return
        registry.delete(name)
        self._refresh()
        self.hint.set_text(f"Deleted '{name}'.")
