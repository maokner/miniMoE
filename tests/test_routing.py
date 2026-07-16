import torch

from model import Model, ModelConfig, MoEFeedForward
from routing import RoutingCollector


class Capture:
    def __init__(self):
        self.calls = []

    def __call__(self, *values):
        self.calls.append(values)


def configured_moe():
    moe = MoEFeedForward(8, 16, num_experts=8, top_k=2).eval()
    with torch.no_grad():
        moe.router.weight.zero_()
        for expert in range(8):
            moe.router.weight[expert].fill_(8 - expert)
    return moe


def observed_routes(mode):
    moe = configured_moe()
    capture = Capture()
    moe.set_routing(mode, capture)
    torch.manual_seed(67)
    moe(torch.ones(2, 3, 8), token_ids=torch.arange(6).reshape(2, 3))
    return capture.calls[0]


def test_learned_and_random_top2_have_two_unique_normalized_routes():
    for mode in ("learned_top2", "random_top2"):
        _, _, indices, weights, _ = observed_routes(mode)
        assert indices.shape[-1] == 2
        assert torch.all(indices[..., 0] != indices[..., 1])
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights[..., 0])
        )


def test_top1_uniform_and_ablation_behavior():
    _, _, top1_indices, top1_weights, _ = observed_routes("learned_top1")
    assert top1_indices.shape[-1] == 1
    assert torch.all(top1_indices == 0)
    assert torch.all(top1_weights == 1)

    _, _, uniform_indices, uniform_weights, _ = observed_routes("uniform_all")
    assert uniform_indices.shape[-1] == 8
    assert torch.equal(uniform_indices[0, 0], torch.arange(8))
    torch.testing.assert_close(uniform_weights.sum(dim=-1), torch.ones(2, 3))

    _, _, ablated_indices, ablated_weights, _ = observed_routes("ablate_expert_0")
    assert torch.all(ablated_weights[ablated_indices == 0] == 0)
    torch.testing.assert_close(ablated_weights.sum(dim=-1), torch.ones(2, 3))
    selected = ablated_indices.gather(-1, ablated_weights.argmax(dim=-1, keepdim=True))
    assert torch.all(selected == 1)


def test_observer_token_totals_match_valid_processed_tokens():
    config = ModelConfig(
        max_seq_length=12,
        vocab_size=32,
        num_layers=2,
        hidden_dim=8,
        num_experts=8,
        top_k=2,
    )
    model = Model(config).eval()
    collector = RoutingCollector(config.num_layers, config.num_experts)
    model.set_routing("learned_top2", collector)
    tokens = torch.randint(0, config.vocab_size, (2, 5))
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    collector.expect_mask(valid)
    model(tokens)
    summary = collector.summary()
    assert [layer["tokens"] for layer in summary["layers"]] == [8, 8]
    assert [sum(layer["route_counts"]) for layer in summary["layers"]] == [16, 16]
