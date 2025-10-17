from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..core.config import settings


@dataclass
class DatasetMetadata:
    name: str
    document_count: int
    chunk_count: int
    last_ingested_at: str


class DatasetRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._cache: Dict[str, DatasetMetadata] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._cache = {
                item["name"]: DatasetMetadata(**item) for item in data  # type: ignore[arg-type]
            }
        else:
            self._cache = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(metadata) for metadata in self._cache.values()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update(self, name: str, document_count: int, chunk_count: int) -> DatasetMetadata:
        metadata = DatasetMetadata(
            name=name,
            document_count=document_count,
            chunk_count=chunk_count,
            last_ingested_at=datetime.utcnow().isoformat(),
        )
        self._cache[name] = metadata
        self._save()
        return metadata

    def list(self) -> List[DatasetMetadata]:
        return list(self._cache.values())

    def get(self, name: str) -> DatasetMetadata | None:
        return self._cache.get(name)


dataset_registry = DatasetRegistry(settings.data_dir / "datasets.json")
