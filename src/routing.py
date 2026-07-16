"""Detached routing telemetry aggregation for evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch


class RoutingCollector:
    def __init__(self, num_layers: int, num_experts: int, capture_tokens: bool = False):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.capture_tokens = capture_tokens
        self.valid_masks: list[torch.Tensor] = []
        self.utilization = torch.zeros(num_layers, num_experts, dtype=torch.float64)
        self.weight_sums = torch.zeros_like(self.utilization)
        self.probability_sums = torch.zeros_like(self.utilization)
        self.entropy_sums = torch.zeros(num_layers, dtype=torch.float64)
        self.tokens = torch.zeros(num_layers, dtype=torch.long)
        self.token_routes: list[dict[str, Any]] = []

    def expect_mask(self, mask: torch.Tensor) -> None:
        """Set one validity mask consumed by each layer on the next forward."""
        self.valid_masks = [mask.detach().bool().cpu() for _ in range(self.num_layers)]

    def __call__(self, layer, token_ids, expert_indices, routing_weights, router_probs):
        token_ids = token_ids.detach().cpu()
        expert_indices = expert_indices.detach().cpu()
        routing_weights = routing_weights.detach().float().cpu()
        router_probs = router_probs.detach().float().cpu()
        valid = (
            self.valid_masks[layer]
            if self.valid_masks
            else torch.ones_like(token_ids, dtype=torch.bool)
        )
        flat_valid = valid.reshape(-1)
        flat_ids = token_ids.reshape(-1)[flat_valid]
        indices = expert_indices.reshape(-1, expert_indices.size(-1))[flat_valid]
        weights = routing_weights.reshape(-1, routing_weights.size(-1))[flat_valid]
        probabilities = router_probs.reshape(-1, router_probs.size(-1))[flat_valid]

        self.tokens[layer] += len(flat_ids)
        self.probability_sums[layer] += probabilities.sum(dim=0, dtype=torch.float64)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        self.entropy_sums[layer] += entropy.sum(dtype=torch.float64)
        for slot in range(indices.size(-1)):
            selected = indices[:, slot]
            selected_weights = weights[:, slot]
            self.utilization[layer].scatter_add_(
                0, selected, (selected_weights > 0).double()
            )
            self.weight_sums[layer].scatter_add_(0, selected, selected_weights.double())

        if self.capture_tokens:
            for token, selected, selected_weights, probabilities_row in zip(
                flat_ids.tolist(),
                indices.tolist(),
                weights.tolist(),
                probabilities.tolist(),
            ):
                self.token_routes.append(
                    {
                        "layer": int(layer),
                        "token_id": int(token),
                        "experts": [int(value) for value in selected],
                        "weights": selected_weights,
                        "router_probabilities": probabilities_row,
                    }
                )

    def summary(self) -> dict[str, Any]:
        layers = []
        for layer in range(self.num_layers):
            token_count = int(self.tokens[layer])
            utilization = self.utilization[layer].tolist()
            total_routes = sum(utilization)
            shares = [
                value / total_routes if total_routes else 0.0 for value in utilization
            ]
            mean_share = 1.0 / self.num_experts
            coefficient_of_variation = (
                math.sqrt(
                    sum((share - mean_share) ** 2 for share in shares)
                    / self.num_experts
                )
                / mean_share
                if total_routes
                else None
            )
            layers.append(
                {
                    "layer": layer,
                    "tokens": token_count,
                    "route_counts": utilization,
                    "utilization": shares,
                    "routing_weight_sums": self.weight_sums[layer].tolist(),
                    "mean_router_probabilities": (
                        (self.probability_sums[layer] / token_count).tolist()
                        if token_count
                        else None
                    ),
                    "mean_router_entropy": (
                        float(self.entropy_sums[layer] / token_count)
                        if token_count
                        else None
                    ),
                    "load_balance_cv": coefficient_of_variation,
                }
            )
        return {"layers": layers, "token_routes": self.token_routes}


def merge_routing_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"layers": [], "token_routes": []}
    num_layers = len(summaries[0]["layers"])
    layers = []
    for layer_index in range(num_layers):
        source_layers = [summary["layers"][layer_index] for summary in summaries]
        tokens = sum(layer["tokens"] for layer in source_layers)
        counts = [
            sum(layer["route_counts"][i] for layer in source_layers)
            for i in range(len(source_layers[0]["route_counts"]))
        ]
        weight_sums = [
            sum(layer["routing_weight_sums"][i] for layer in source_layers)
            for i in range(len(counts))
        ]
        entropy_sum = sum(
            layer["mean_router_entropy"] * layer["tokens"]
            for layer in source_layers
            if layer["tokens"]
        )
        probability_sums = [
            sum(
                (layer["mean_router_probabilities"] or [0] * len(counts))[i]
                * layer["tokens"]
                for layer in source_layers
            )
            for i in range(len(counts))
        ]
        total_routes = sum(counts)
        utilization = [
            count / total_routes if total_routes else 0.0 for count in counts
        ]
        mean = 1.0 / len(counts)
        cv = (
            math.sqrt(sum((value - mean) ** 2 for value in utilization) / len(counts))
            / mean
            if total_routes
            else None
        )
        layers.append(
            {
                "layer": layer_index,
                "tokens": tokens,
                "route_counts": counts,
                "utilization": utilization,
                "routing_weight_sums": weight_sums,
                "mean_router_probabilities": [
                    value / tokens for value in probability_sums
                ]
                if tokens
                else None,
                "mean_router_entropy": entropy_sum / tokens if tokens else None,
                "load_balance_cv": cv,
            }
        )
    routes = [
        route for summary in summaries for route in summary.get("token_routes", [])
    ]
    return {"layers": layers, "token_routes": routes}
