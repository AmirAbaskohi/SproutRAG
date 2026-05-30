from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from sproutrag.data.schema import DocumentIndex, TreeNode
from sproutrag.retrieval.schema import (
    RetrievalResult,
    retrieval_result_from_node,
    sort_retrieval_results,
)
from sproutrag.retrieval.scoring import score_node


class HierarchicalRetriever:
    def __init__(
        self,
        encoder: Any | None = None,
        collect_strategy: str = "threshold",
        include_root: bool = False,
        reranker: Any | None = None,
    ) -> None:
        if encoder is not None and not hasattr(encoder, "encode_query"):
            raise ValueError("encoder must implement encode_query")
        if collect_strategy not in {"threshold", "visited"}:
            raise ValueError("collect_strategy must be 'threshold' or 'visited'")
        if not isinstance(include_root, bool):
            raise ValueError("include_root must be a boolean")
        if reranker is not None and not hasattr(reranker, "rerank"):
            raise ValueError("reranker must implement rerank")
        self.encoder = encoder
        self.collect_strategy = collect_strategy
        self.include_root = include_root
        self.reranker = reranker

    def _validate_index(self, index: DocumentIndex) -> None:
        if not isinstance(index, DocumentIndex):
            raise ValueError("index must be a DocumentIndex")
        if index.root_id is None:
            raise ValueError("index.root_id must not be None")
        if index.root_id not in index.nodes:
            raise ValueError("index.root_id must exist in index.nodes")

    def _should_collect(self, score: float, threshold: float) -> bool:
        if self.collect_strategy == "visited":
            return True
        return score >= threshold

    def _collect_result(
        self,
        node: TreeNode,
        score: float,
        beam_width: int,
        threshold: float,
        fallback: bool = False,
        fallback_reason: str | None = None,
    ) -> RetrievalResult:
        metadata = {
            "retrieval_score": float(score),
            "retrieval_strategy": "hierarchical_beam_search",
            "collect_strategy": self.collect_strategy,
            "beam_width": beam_width,
            "threshold": float(threshold),
            "visited": True,
        }
        if fallback:
            metadata["fallback"] = True
            if fallback_reason is not None:
                metadata["fallback_reason"] = fallback_reason
        return retrieval_result_from_node(node, score, extra_metadata=metadata)

    def _sort_beam_candidates(
        self,
        candidates: list[tuple[TreeNode, float]],
        beam_width: int,
    ) -> list[TreeNode]:
        def key(item: tuple[TreeNode, float]) -> tuple[float, int, str]:
            node, score = item
            return (-score, node.depth, node.node_id)

        sorted_items = sorted(candidates, key=key)
        return [node for node, _ in sorted_items[:beam_width]]

    def retrieve(
        self,
        query: str,
        index: DocumentIndex,
        top_k: int = 5,
        beam_width: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        if self.encoder is None:
            raise ValueError("encoder must be provided to retrieve using query string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        embedding = self.encoder.encode_query(query)
        results = self.retrieve_from_embedding(
            embedding, index, top_k=top_k, beam_width=beam_width, threshold=threshold
        )
        if self.reranker is not None:
            return self.reranker.rerank(query, results, top_k=top_k)
        return results

    def retrieve_from_embedding(
        self,
        query_embedding: torch.Tensor,
        index: DocumentIndex,
        top_k: int = 5,
        beam_width: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        if not isinstance(query_embedding, torch.Tensor):
            raise ValueError("query_embedding must be a torch.Tensor")
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be 1-dimensional")
        if not torch.isfinite(query_embedding).all():
            raise ValueError("query_embedding must contain finite values")
        self._validate_index(index)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(beam_width, int) or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError("threshold must be a finite number")

        root = index.nodes[index.root_id]
        beam = [root]
        visited_scores: dict[str, float] = {}
        collected: dict[str, RetrievalResult] = {}

        if self.include_root:
            root_score = score_node(query_embedding, root)
            visited_scores[root.node_id] = root_score
            if self._should_collect(root_score, threshold):
                collected[root.node_id] = self._collect_result(
                    root, root_score, beam_width, threshold
                )

        while beam:
            candidates: dict[str, TreeNode] = {}
            for node in beam:
                for child_id in node.children:
                    if child_id not in index.nodes:
                        raise ValueError(f"child node {child_id} missing from index")
                    if child_id not in candidates:
                        candidates[child_id] = index.nodes[child_id]
            if not candidates:
                break

            candidate_scores: list[tuple[TreeNode, float]] = []
            for node_id, node in candidates.items():
                score = score_node(query_embedding, node)
                visited_scores[node_id] = score
                candidate_scores.append((node, score))
                if self._should_collect(score, threshold):
                    collected[node_id] = self._collect_result(
                        node, score, beam_width, threshold
                    )

            beam = self._sort_beam_candidates(candidate_scores, beam_width)

        if not collected:
            if not visited_scores and not self.include_root:
                root_score = score_node(query_embedding, root)
                visited_scores[root.node_id] = root_score
            best = sorted(
                visited_scores.items(), key=lambda item: (-item[1], item[0])
            )
            for node_id, score in best:
                node = index.nodes[node_id]
                collected[node_id] = self._collect_result(
                    node,
                    score,
                    beam_width,
                    threshold,
                    fallback=True,
                    fallback_reason="no_candidates_met_threshold",
                )

        return sort_retrieval_results(list(collected.values()), top_k=top_k)
