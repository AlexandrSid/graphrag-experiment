from __future__ import annotations

import sys


def log(message: str) -> None:
    print(message, flush=True)
    sys.stdout.flush()
