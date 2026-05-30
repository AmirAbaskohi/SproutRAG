from __future__ import annotations

import re
from typing import Iterable

from .schema import DocumentIndex, RawDocument, SentenceChunk, TreeNode


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _trimmed_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    slice_text = text[start:end]
    lstrip_len = len(slice_text) - len(slice_text.lstrip())
    rstrip_len = len(slice_text) - len(slice_text.rstrip())
    trimmed_start = start + lstrip_len
    trimmed_end = end - rstrip_len
    trimmed_text = text[trimmed_start:trimmed_end]
    return trimmed_text, trimmed_start, trimmed_end


def _fallback_split_sentences(text: str) -> list[tuple[str, int, int]]:
    sentences: list[tuple[str, int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.start()
        trimmed_text, trimmed_start, trimmed_end = _trimmed_span(text, start, end)
        if trimmed_text:
            sentences.append((trimmed_text, trimmed_start, trimmed_end))
        start = match.end()
    if start <= len(text):
        trimmed_text, trimmed_start, trimmed_end = _trimmed_span(text, start, len(text))
        if trimmed_text:
            sentences.append((trimmed_text, trimmed_start, trimmed_end))
    return sentences


def _spacy_split_sentences(text: str) -> list[tuple[str, int, int]]:
    import spacy  # type: ignore

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences: list[tuple[str, int, int]] = []
    for sent in doc.sents:
        trimmed_text, trimmed_start, trimmed_end = _trimmed_span(
            text, sent.start_char, sent.end_char
        )
        if trimmed_text:
            sentences.append((trimmed_text, trimmed_start, trimmed_end))
    return sentences


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    try:
        return _spacy_split_sentences(text)
    except Exception:
        return _fallback_split_sentences(text)


def chunk_sentences(
    document: RawDocument,
    max_sentences_per_chunk: int = 2,
) -> list[SentenceChunk]:
    if max_sentences_per_chunk < 1:
        raise ValueError("max_sentences_per_chunk must be >= 1")
    sentences = split_sentences(document.text)
    if not sentences:
        chunk_id = f"{document.doc_id}::chunk::0"
        chunk_text = document.text.strip()
        end_char = len(document.text)
        return [
            SentenceChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=chunk_text,
                start_char=0,
                end_char=end_char if end_char > 0 else 1,
                sentence_ids=[0],
                metadata={"chunk_index": 0, "num_sentences": 1},
            )
        ]

    chunks: list[SentenceChunk] = []
    chunk_index = 0
    for i in range(0, len(sentences), max_sentences_per_chunk):
        group = sentences[i : i + max_sentences_per_chunk]
        start_char = group[0][1]
        end_char = group[-1][2]
        chunk_text = document.text[start_char:end_char].strip()
        sentence_ids = list(range(i, i + len(group)))
        chunk_id = f"{document.doc_id}::chunk::{chunk_index}"
        chunks.append(
            SentenceChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                sentence_ids=sentence_ids,
                metadata={
                    "chunk_index": chunk_index,
                    "num_sentences": len(group),
                },
            )
        )
        chunk_index += 1
    return chunks


def make_leaf_nodes(chunks: list[SentenceChunk]) -> list[TreeNode]:
    nodes: list[TreeNode] = []
    for chunk in chunks:
        chunk_index = chunk.metadata.get("chunk_index")
        node_id = f"{chunk.doc_id}::leaf::{chunk_index}"
        nodes.append(
            TreeNode(
                node_id=node_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                sentence_chunk_ids=[chunk.chunk_id],
                depth=0,
                is_leaf=True,
                metadata={
                    "source_chunk_id": chunk.chunk_id,
                    "chunk_index": chunk_index,
                },
            )
        )
    return nodes


def build_empty_document_index(
    document: RawDocument,
    max_sentences_per_chunk: int = 2,
) -> DocumentIndex:
    chunks = chunk_sentences(document, max_sentences_per_chunk=max_sentences_per_chunk)
    nodes = make_leaf_nodes(chunks)
    metadata = dict(document.metadata)
    if document.title is not None:
        metadata["title"] = document.title
    index = DocumentIndex(
        doc_id=document.doc_id,
        chunks={chunk.chunk_id: chunk for chunk in chunks},
        nodes={node.node_id: node for node in nodes},
        root_id=None,
        metadata=metadata,
    )
    return index
