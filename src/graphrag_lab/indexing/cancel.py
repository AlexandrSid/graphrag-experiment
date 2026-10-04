from __future__ import annotations

import signal
from pathlib import Path

_flag = False
_stop_file: Path | None = None


def bind(index_dir: Path, stage: str) -> None:
    global _flag, _stop_file
    _flag = False
    folder = index_dir / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    _stop_file = folder / f"{stage}.stop"
    if _stop_file.exists():
        _stop_file.unlink()


def unbind() -> None:
    global _flag, _stop_file
    if _stop_file and _stop_file.exists():
        _stop_file.unlink()
    _stop_file = None
    _flag = False


def request_stop() -> None:
    global _flag
    _flag = True
    if _stop_file is not None:
        _stop_file.write_text("stop", encoding="utf-8")


def write_stop(index_dir: Path, stage: str) -> Path:
    path = index_dir / "locks" / f"{stage}.stop"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stop", encoding="utf-8")
    return path


def stopped() -> bool:
    if _flag:
        return True
    return bool(_stop_file and _stop_file.exists())


def check() -> None:
    if stopped():
        raise KeyboardInterrupt("stage stop requested")


def install_handlers() -> None:
    signal.signal(signal.SIGINT, lambda *_args: request_stop())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_args: request_stop())
