# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Allow ``python -m pdeobs`` to behave like the console command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
