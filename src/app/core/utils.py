from __future__ import annotations

import glob
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, List


def resolve_paths(paths: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        expanded = glob.glob(path)
        if not expanded:
            resolved.append(Path(path).expanduser())
        else:
            resolved.extend(Path(p).expanduser() for p in expanded)
    unique = []
    seen = set()
    for path in resolved:
        normalized = path.resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def dict_hash(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@contextmanager
def time_block() -> Generator[float, None, None]:
    start = time.perf_counter()
    yield start
    end = time.perf_counter()
    duration = end - start
    # Communicate duration via attribute to allow optional retrieval.
    setattr(time_block, "last_duration", duration)  # type: ignore[attr-defined]


def ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
