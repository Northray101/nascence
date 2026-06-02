"""Entry point: ``python -m nascence`` (or ``bash run.sh``)."""

from __future__ import annotations


def main() -> None:
    from .ui.app import AppController

    AppController().run()


if __name__ == "__main__":
    main()
