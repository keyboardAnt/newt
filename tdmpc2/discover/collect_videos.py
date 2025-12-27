#!/usr/bin/env python3
"""Deprecated: use the canonical `python -m discover videos collect` command.

This file previously implemented its own log/video scanning logic, which caused
discrepancies vs the `discover` CLI. We intentionally keep this stub so old
references fail loudly and point users to the correct command.
"""

from __future__ import annotations

raise SystemExit(
    "This script is deprecated.\n\n"
    "Use:\n"
    "  python -m discover videos collect --min-progress 0.5\n"
    "Options:\n"
    "  --symlink   (default is copy)\n"
    "  --output <dir>\n"
)
