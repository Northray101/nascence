"""Species manager: create, list, train and delete named species."""

from __future__ import annotations

import pygame
import pygame_gui
from pygame_gui.elements import (
    UIButton,
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
            rect=pygame.Rect((0, 0), (380, 280)),
            manager=manager,
            window_display_title="New species",
            object_id="#new_species",
        )
        self.set_blocking(True)
        w = 340

        UILabel(
            relative_rect=pygame.Rect((10, 10), (w, 26)),
            text="Name your bacteria:",
            manager=manager,
            container=self,
        )
        self.name_entry = UITextEntryLine(
            relative_rect=pygame.Rect((10, 40), (w, 36)),
            manager=manager,
            container=self,
        )
        self.name_entry.set_text("Wiggler")

        self.legs_label = UILabel(
            relative_rect=pygame.Rect((10, 90), (w, 26)),
            text="Legs: 3",
            manager=manager,
            container=self,
        )
        self.legs_slider = UIHorizontalSlider(
            relative_rect=pygame.Rect((10, 118), (w, 28)),
            start_value=3,
            value_range=(2, 6),
            manager=manager,
            container=self,
        )

        self.create_btn = UIButton(
            relative_rect=pygame.Rect((10, 200), (160, 40)),
            text="Create",
            manager=manager,
            container=self,
        )
        self.cancel_btn = UIButton(
            relative_rect=pygame.Rect((190, 200), (160, 40)),
            text="Cancel",
            manager=manager,
            container=self,
        )

    @property
    def leg_count(self) -> int:
        return int(round(self.legs_slider.get_current_value()))


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
            if self.dialog is not None and event.ui_element == self.dialog.legs_slider:
                self.dialog.legs_label.set_text(f"Legs: {self.dialog.leg_count}")

        elif event.type == pygame_gui.UI_WINDOW_CLOSE:
            if self.dialog is not None and event.ui_element == self.dialog:
                self.dialog = None

    # -- actions ------------------------------------------------------------
    def _create_from_dialog(self) -> None:
        assert self.dialog is not None
        name = self.dialog.name_entry.get_text().strip() or "Wiggler"
        if registry.exists(name):
            self.hint.set_text("Name already exists.")
            return
        morph = CreatureMorphology(num_legs=self.dialog.leg_count)
        registry.create(name, morph)
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
