from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from sproutrag.data.schema import DocumentIndex, SentenceChunk, TreeNode
from sproutrag.tree.utils import (
    concatenate_child_text,
    make_internal_node_id,
    make_leaf_node_id,
    tensor_to_float_list,
    validate_tree_inputs,
)


class AttentionTreeBuilder:
    def __init__(
        self,
        linkage: str = "max",
        preserve_input_order_text: bool = True,
    ) -> None:
        if linkage != "max":
            raise ValueError("linkage must be 'max'")
        if not isinstance(preserve_input_order_text, bool):
            raise ValueError("preserve_input_order_text must be a boolean")
        self.linkage = linkage
        self.preserve_input_order_text = preserve_input_order_text

    def _select_best_pair(
        self,
        active_node_ids: list[str],
        active_relation: torch.Tensor,
        active_min_chunk_index: dict[str, int],
        active_max_chunk_index: dict[str, int],
    ) -> tuple[int, int, float]:
        best = None
        best_tuple = None
        for i in range(len(active_node_ids)):
            for j in range(i + 1, len(active_node_ids)):
                score_tensor = active_relation[i, j]
                score = float(score_tensor.detach().cpu())
                if not torch.isfinite(score_tensor).item():
                    continue
                id_i = active_node_ids[i]
                id_j = active_node_ids[j]
                candidate = (
                    -score,
                    min(active_min_chunk_index[id_i], active_min_chunk_index[id_j]),
                    max(active_max_chunk_index[id_i], active_max_chunk_index[id_j]),
                    min(id_i, id_j),
                    max(id_i, id_j),
                    i,
                    j,
                )
                if best_tuple is None or candidate < best_tuple:
                    best_tuple = candidate
                    best = (i, j, score)
        if best is None:
            raise ValueError("no valid pair found for merge")
        return best

    def _rebuild_relation_matrix_after_merge(
        self,
        active_node_ids_before: list[str],
        active_relation_before: torch.Tensor,
        left_id: str,
        right_id: str,
        parent_id: str,
        active_node_ids_after: list[str],
    ) -> torch.Tensor:
        device = active_relation_before.device
        dtype = active_relation_before.dtype
        size = len(active_node_ids_after)
        new_relation = torch.zeros((size, size), device=device, dtype=dtype)
        index_before = {node_id: idx for idx, node_id in enumerate(active_node_ids_before)}
        left_idx = index_before[left_id]
        right_idx = index_before[right_id]

        for i, node_i in enumerate(active_node_ids_after):
            for j, node_j in enumerate(active_node_ids_after):
                if i == j:
                    new_relation[i, j] = 0
                    continue
                if node_i == parent_id and node_j != parent_id:
                    old_j = index_before[node_j]
                    value = torch.maximum(
                        active_relation_before[left_idx, old_j],
                        active_relation_before[right_idx, old_j],
                    )
                    new_relation[i, j] = value
                elif node_j == parent_id and node_i != parent_id:
                    old_i = index_before[node_i]
                    value = torch.maximum(
                        active_relation_before[old_i, left_idx],
                        active_relation_before[old_i, right_idx],
                    )
                    new_relation[i, j] = value
                else:
                    old_i = index_before[node_i]
                    old_j = index_before[node_j]
                    new_relation[i, j] = active_relation_before[old_i, old_j]
        return new_relation

    def build_tree(
        self,
        doc_id: str,
        chunks: list[SentenceChunk],
        embeddings: torch.Tensor,
        mutual_attention: torch.Tensor,
    ) -> DocumentIndex:
        validate_tree_inputs(doc_id, chunks, embeddings, mutual_attention)

        index = DocumentIndex(doc_id=doc_id)
        for chunk in chunks:
            index.add_chunk(chunk)

        active_node_ids: list[str] = []
        active_embeddings: dict[str, torch.Tensor] = {}
        active_sentence_chunk_ids: dict[str, list[str]] = {}
        active_min_chunk_index: dict[str, int] = {}
        active_max_chunk_index: dict[str, int] = {}

        for idx, chunk in enumerate(chunks):
            chunk_index = chunk.metadata.get("chunk_index", idx)
            node_id = make_leaf_node_id(doc_id, chunk_index)
            leaf = TreeNode(
                node_id=node_id,
                doc_id=doc_id,
                text=chunk.text,
                embedding=tensor_to_float_list(embeddings[idx]),
                left=None,
                right=None,
                parent=None,
                children=[],
                sentence_chunk_ids=[chunk.chunk_id],
                depth=0,
                is_leaf=True,
                metadata={
                    "source_chunk_id": chunk.chunk_id,
                    "chunk_index": chunk_index,
                    "node_type": "leaf",
                },
            )
            index.add_node(leaf)
            active_node_ids.append(node_id)
            active_embeddings[node_id] = embeddings[idx]
            active_sentence_chunk_ids[node_id] = [chunk.chunk_id]
            active_min_chunk_index[node_id] = int(chunk_index)
            active_max_chunk_index[node_id] = int(chunk_index)

        if len(chunks) == 1:
            index.root_id = active_node_ids[0]
            index.metadata.update(
                {
                    "num_chunks": 1,
                    "num_nodes": 1,
                    "root_id": index.root_id,
                    "linkage": self.linkage,
                    "preserve_input_order_text": self.preserve_input_order_text,
                }
            )
            return index

        active_relation = mutual_attention.clone()
        merge_index = 0

        while len(active_node_ids) > 1:
            left_idx, right_idx, score = self._select_best_pair(
                active_node_ids,
                active_relation,
                active_min_chunk_index,
                active_max_chunk_index,
            )
            u_id = active_node_ids[left_idx]
            v_id = active_node_ids[right_idx]

            if self.preserve_input_order_text:
                if active_min_chunk_index[u_id] <= active_min_chunk_index[v_id]:
                    left_id, right_id = u_id, v_id
                else:
                    left_id, right_id = v_id, u_id
            else:
                left_id, right_id = u_id, v_id

            parent_id = make_internal_node_id(doc_id, merge_index)
            left_emb = active_embeddings[left_id]
            right_emb = active_embeddings[right_id]
            parent_embedding = (left_emb + right_emb) / 2.0

            if self.preserve_input_order_text:
                if active_min_chunk_index[left_id] <= active_min_chunk_index[right_id]:
                    parent_sentence_chunk_ids = (
                        active_sentence_chunk_ids[left_id]
                        + active_sentence_chunk_ids[right_id]
                    )
                else:
                    parent_sentence_chunk_ids = (
                        active_sentence_chunk_ids[right_id]
                        + active_sentence_chunk_ids[left_id]
                    )
            else:
                parent_sentence_chunk_ids = (
                    active_sentence_chunk_ids[left_id]
                    + active_sentence_chunk_ids[right_id]
                )

            left_node = index.nodes[left_id]
            right_node = index.nodes[right_id]
            parent_text = concatenate_child_text(left_node.text, right_node.text)
            parent_depth = 1 + max(left_node.depth, right_node.depth)
            parent_min_chunk_index = min(
                active_min_chunk_index[left_id], active_min_chunk_index[right_id]
            )
            parent_max_chunk_index = max(
                active_max_chunk_index[left_id], active_max_chunk_index[right_id]
            )

            parent_node = TreeNode(
                node_id=parent_id,
                doc_id=doc_id,
                text=parent_text,
                embedding=tensor_to_float_list(parent_embedding),
                left=left_id,
                right=right_id,
                parent=None,
                children=[left_id, right_id],
                sentence_chunk_ids=parent_sentence_chunk_ids,
                depth=parent_depth,
                is_leaf=False,
                metadata={
                    "node_type": "internal",
                    "merge_index": merge_index,
                    "left_child": left_id,
                    "right_child": right_id,
                    "num_sentence_chunks": len(parent_sentence_chunk_ids),
                    "min_chunk_index": parent_min_chunk_index,
                    "max_chunk_index": parent_max_chunk_index,
                    "linkage": self.linkage,
                    "merge_score": float(score),
                },
            )
            index.add_node(parent_node)
            index.nodes[left_id].parent = parent_id
            index.nodes[right_id].parent = parent_id

            active_node_ids_before = list(active_node_ids)
            active_node_ids = [
                node_id
                for node_id in active_node_ids
                if node_id not in {left_id, right_id}
            ]
            active_node_ids.append(parent_id)

            active_embeddings.pop(left_id)
            active_embeddings.pop(right_id)
            active_embeddings[parent_id] = parent_embedding

            active_sentence_chunk_ids.pop(left_id)
            active_sentence_chunk_ids.pop(right_id)
            active_sentence_chunk_ids[parent_id] = parent_sentence_chunk_ids

            active_min_chunk_index.pop(left_id)
            active_min_chunk_index.pop(right_id)
            active_min_chunk_index[parent_id] = parent_min_chunk_index

            active_max_chunk_index.pop(left_id)
            active_max_chunk_index.pop(right_id)
            active_max_chunk_index[parent_id] = parent_max_chunk_index

            active_relation = self._rebuild_relation_matrix_after_merge(
                active_node_ids_before,
                active_relation,
                left_id,
                right_id,
                parent_id,
                active_node_ids,
            )

            merge_index += 1

        index.root_id = active_node_ids[0]
        index.metadata.update(
            {
                "num_chunks": len(chunks),
                "num_nodes": 2 * len(chunks) - 1,
                "root_id": index.root_id,
                "linkage": self.linkage,
                "preserve_input_order_text": self.preserve_input_order_text,
            }
        )
        return index
