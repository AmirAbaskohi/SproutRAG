from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_str_list(value: list[str], field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")


@dataclass
class IndexManifest:
    index_name: str
    created_at: str
    num_documents: int
    doc_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.index_name, "index_name")
        _require_non_empty_str(self.created_at, "created_at")
        if not isinstance(self.num_documents, int) or self.num_documents < 0:
            raise ValueError("num_documents must be an integer >= 0")
        _require_str_list(self.doc_ids, "doc_ids")
        if len(self.doc_ids) != self.num_documents:
            raise ValueError("doc_ids length must match num_documents")
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("doc_ids must not contain duplicates")
        _require_dict(self.metadata, "metadata")


def make_manifest(
    index_name: str,
    doc_ids: list[str],
    metadata: dict[str, Any] | None = None,
) -> IndexManifest:
    _require_non_empty_str(index_name, "index_name")
    _require_str_list(doc_ids, "doc_ids")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("doc_ids must not contain duplicates")
    created_at = datetime.now(timezone.utc).isoformat()
    return IndexManifest(
        index_name=index_name,
        created_at=created_at,
        num_documents=len(doc_ids),
        doc_ids=list(doc_ids),
        metadata={} if metadata is None else dict(metadata),
    )


def manifest_to_dict(manifest: IndexManifest) -> dict[str, Any]:
    if not isinstance(manifest, IndexManifest):
        raise ValueError("manifest must be an IndexManifest")
    return {
        "index_name": manifest.index_name,
        "created_at": manifest.created_at,
        "num_documents": manifest.num_documents,
        "doc_ids": list(manifest.doc_ids),
        "metadata": dict(manifest.metadata),
    }


def manifest_from_dict(data: dict[str, Any]) -> IndexManifest:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    return IndexManifest(
        index_name=data["index_name"],
        created_at=data["created_at"],
        num_documents=data["num_documents"],
        doc_ids=list(data["doc_ids"]),
        metadata=data.get("metadata", {}),
    )


def save_manifest(manifest: IndexManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest_to_dict(manifest)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)


def load_manifest(path: str | Path) -> IndexManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return manifest_from_dict(data)
