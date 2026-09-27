"""Точка входа пакета: ``python -m app.story_pipeline``.

Тонкая обёртка: вся логика и коды возврата живут в
`app.story_pipeline.cli.main`.
"""

from __future__ import annotations

import sys

from app.story_pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
