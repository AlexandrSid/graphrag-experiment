from __future__ import annotations

import os
from pathlib import Path


def lock_path(index_dir: Path, stage: str) -> Path:
    folder = index_dir / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stage}.pid"


def write_lock(index_dir: Path, stage: str) -> None:
    lock_path(index_dir, stage).write_text(str(os.getpid()), encoding="utf-8")


def clear_lock(index_dir: Path, stage: str) -> None:
    path = lock_path(index_dir, stage)
    if path.exists():
        path.unlink()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_pid(index_dir: Path, stage: str) -> int | None:
    path = lock_path(index_dir, stage)
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def owner_alive(index_dir: Path, stage: str) -> bool:
    pid = lock_pid(index_dir, stage)
    if pid is None:
        return False
    return pid_alive(pid)


def reconcile_running(store, index_dir: Path) -> list[str]:
    changed: list[str] = []
    for row in store.all_stages():
        if row["status"] != "running":
            continue
        name = row["name"]
        if owner_alive(index_dir, name):
            continue
        store.set_stage(name, "interrupted")
        clear_lock(index_dir, name)
        changed.append(name)
    return changed
