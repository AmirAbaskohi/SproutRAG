from __future__ import annotations

from pathlib import Path
from typing import Any

from sproutrag.data.schema import DocumentIndex
from sproutrag.data.serialization import (
    load_document_index as load_single_document_index,
    save_document_index as save_single_document_index,
)
from sproutrag.indexing.manifest import (
    IndexManifest,
    load_manifest as load_index_manifest,
    make_manifest,
    save_manifest,
)


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _safe_doc_id(doc_id: str) -> str:
    return (
        doc_id.replace("/", "__slash__")
        .replace("\\", "__backslash__")
        .replace(":", "__colon__")
        .replace(" ", "_")
    )


class IndexStore:
    def __init__(self, root_dir: str | Path, index_name: str = "default") -> None:
        if not isinstance(root_dir, (str, Path)):
            raise ValueError("root_dir must be a string or Path")
        _require_non_empty_str(index_name, "index_name")
        self.root_dir = Path(root_dir)
        self.index_name = index_name
        self.index_dir = self.root_dir / index_name
        self.documents_dir = self.index_dir / "documents"
        self.manifest_path = self.index_dir / "manifest.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def document_path(self, doc_id: str) -> Path:
        _require_non_empty_str(doc_id, "doc_id")
        safe_doc_id = _safe_doc_id(doc_id)
        return self.documents_dir / f"{safe_doc_id}.json"

    def save_document_index(self, index: DocumentIndex) -> Path:
        if not isinstance(index, DocumentIndex):
            raise ValueError("index must be a DocumentIndex")
        path = self.document_path(index.doc_id)
        save_single_document_index(index, path)
        return path

    def load_document_index(self, doc_id: str) -> DocumentIndex:
        _require_non_empty_str(doc_id, "doc_id")
        path = self.document_path(doc_id)
        if not path.exists():
            raise FileNotFoundError(f"document index not found: {path}")
        return load_single_document_index(path)

    def list_document_paths(self) -> list[Path]:
        if not self.documents_dir.exists():
            return []
        return sorted(self.documents_dir.glob("*.json"), key=lambda p: p.name)

    def save_many(
        self,
        indexes: list[DocumentIndex],
        metadata: dict[str, Any] | None = None,
    ) -> IndexManifest:
        if not isinstance(indexes, list):
            raise ValueError("indexes must be a list of DocumentIndex")
        if not all(isinstance(index, DocumentIndex) for index in indexes):
            raise ValueError("indexes must contain DocumentIndex instances")
        doc_ids = [index.doc_id for index in indexes]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("indexes contain duplicate doc_ids")

        for index in indexes:
            self.save_document_index(index)

        manifest_metadata = {
            "index_name": self.index_name,
            "num_saved_documents": len(indexes),
        }
        if metadata:
            manifest_metadata.update(metadata)
        manifest = make_manifest(
            index_name=self.index_name, doc_ids=doc_ids, metadata=manifest_metadata
        )
        save_manifest(manifest, self.manifest_path)
        return manifest

    def load_manifest(self) -> IndexManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        return load_index_manifest(self.manifest_path)

    def load_all(self) -> list[DocumentIndex]:
        if self.manifest_path.exists():
            manifest = self.load_manifest()
            return [self.load_document_index(doc_id) for doc_id in manifest.doc_ids]
        return [load_single_document_index(path) for path in self.list_document_paths()]

    def exists(self, doc_id: str) -> bool:
        return self.document_path(doc_id).exists()

    def clear(self) -> None:
        if self.documents_dir.exists():
            for path in self.documents_dir.glob("*.json"):
                path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
