# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from dataclasses import replace
from itertools import permutations
from unittest.mock import patch

import pytest
import torch

from nemo_rl.data.exact_calls import (
    _build_exact_call_tree,
    _exact_call_trie_diagnostics,
    _flatten_exact_call,
    _page_aligned_execution_prefix_length,
    _tensor_parts_common_prefix_length,
    _tensor_parts_first_row_containing,
)


def _call(tokens: list[int], routes: list[int]) -> list[dict]:
    metadata = {
        "ng_generation_replica_id": "replica-0",
        "ng_generation_weight_version": 0,
        "ng_generation_weight_version_end": 0,
        "ng_kv_cache_scheduler_block_size": 2,
        "ng_kv_cache_hash_block_size": 2,
        "ng_kv_cache_num_cached_tokens": 2,
    }
    return [
        {
            "role": "user",
            "token_ids": torch.tensor(tokens[:-1]),
            "routed_experts": torch.tensor(routes[:-1]).reshape(-1, 1, 1),
            **metadata,
        },
        {
            "role": "assistant",
            "token_ids": torch.tensor(tokens[-1:]),
            "generation_logprobs": torch.tensor([-0.5]),
            "routed_experts": torch.tensor(routes[-1:]).reshape(-1, 1, 1),
            **metadata,
        },
    ]


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_terminal_alias_cannot_merge_conflicting_executed_routes(order) -> None:
    calls = [
        _call([1, 2, 3], [0, 0, -1]),
        _call([1, 2, 3, 4], [0, 0, 1, -1]),
        _call([1, 2, 3, 5], [0, 0, 2, -1]),
    ]
    result = _build_exact_call_tree([calls[index] for index in order])
    tokens = torch.cat([message["token_ids"] for message in result.unique_message_log])
    routes = torch.cat(
        [message["routed_experts"] for message in result.unique_message_log]
    )
    targets = torch.cat(
        [message["token_ids"] for message in result.edge_message_log[1:]]
    )
    sources = torch.tensor(result.layout.edge_source_indices)

    # The two executions of token 3 need separate physical nodes, while the
    # original sample of token 3 still contributes just one loss edge.
    assert tokens.numel() == 6
    assert targets.tolist() == [[3, 4, 5][index] for index in order]
    assert tokens[sources].tolist() == [[2, 3, 3][index] for index in order]
    assert routes[sources].flatten().tolist() == [[0, 1, 2][index] for index in order]
    assert (
        sum(
            message["generation_logprobs"].numel()
            for message in result.edge_message_log[1:]
        )
        == 3
    )


def test_terminal_promotion_keeps_compatible_descendants_shared() -> None:
    result = _build_exact_call_tree(
        [
            _call([1, 2, 3], [0, 0, -1]),
            _call([1, 2, 3, 4], [0, 0, 1, -1]),
            _call([1, 2, 3, 5], [0, 0, 1, -1]),
        ]
    )

    assert result.layout.unique_token_count == 5
    assert result.layout.edge_source_indices == (1, 2, 2)


def test_common_prefix_ignores_empty_message_parts() -> None:
    assert (
        _tensor_parts_common_prefix_length(
            [torch.tensor([], dtype=torch.long), torch.tensor([1, 2])],
            [torch.tensor([1]), torch.tensor([], dtype=torch.long), torch.tensor([3])],
        )
        == 1
    )


def test_diagnostic_cache_matches_uncached_pair_classification() -> None:
    rng = random.Random(20261014)
    for _ in range(40):
        calls = []
        for index in range(12):
            length = rng.randrange(2, 28)
            tokens = list(range(length))
            if rng.random() < 0.2:
                tokens[rng.randrange(length)] = 100
            routes = [0] * (length - 1) + [-1]
            if rng.random() < 0.6:
                routes[rng.randrange(length - 1)] = rng.choice([-1, 1, 2])
            call = _flatten_exact_call(_call(tokens, routes), index)
            call = replace(
                call,
                generation_replica_id=rng.choice(["replica-0", "replica-1", None]),
                kv_cache_num_cached_tokens=rng.choice([None, 0, 2, 8, 64]),
                has_incomplete_routes=rng.random() < 0.1,
            )
            if rng.random() < 0.1:
                call = replace(call, routed_expert_parts=None)
            calls.append(call)
        expected = [
            _page_aligned_execution_prefix_length(prefix, descendant)
            for prefix in calls
            for descendant in calls
        ]
        cache = {}
        actual = [
            _page_aligned_execution_prefix_length(
                prefix, descendant, missing_route_rows=cache
            )
            for prefix in calls
            for descendant in calls
        ]
        assert actual == expected
        forks, shared, replicas = _exact_call_trie_diagnostics(calls)
        assert forks == sum(x is not None for x in expected)
        assert shared == sum(x for x in expected if x is not None)
        assert replicas == {
            c.generation_replica_id
            for c in calls
            if c.generation_replica_id is not None
        }


def test_diagnostic_scan_is_bounded_and_cache_is_not_reused_after_mutation() -> None:
    calls = [
        _flatten_exact_call(
            _call(list(range(length)), [0] * (length - 2) + [1, -1]), index
        )
        for index, length in enumerate([5, 7, 9, 11])
    ]
    # Empty messages and a terminal missing-route placeholder are valid views.
    for call in calls:
        call.routed_expert_parts.insert(0, torch.empty((0, 1, 1), dtype=torch.long))
    with patch(
        "nemo_rl.data.exact_calls._tensor_parts_first_row_containing",
        wraps=_tensor_parts_first_row_containing,
    ) as scan:
        before = _exact_call_trie_diagnostics(calls)
        assert before[0] == 6
        assert scan.call_count == len(calls)
    for call in calls:
        call.routed_expert_parts[1][0] = -1
    assert _exact_call_trie_diagnostics(calls)[:2] == (0, 0)
