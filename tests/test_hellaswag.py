import torch

from experiments import _valid_choice_mask
from hellaswag import collate_examples, completion_losses, score_batch, wilson_interval


class LookupModel(torch.nn.Module):
    def __init__(self, vocab_size=16):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, tokens):
        logits = torch.zeros(*tokens.shape, self.vocab_size)
        targets = tokens[:, 1:]
        logits[:, :-1].scatter_(2, targets.unsqueeze(-1), 2.0)
        return logits, None


def rendered(lengths, label):
    maximum = max(lengths)
    tokens = torch.zeros(4, maximum, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for row, length in enumerate(lengths):
        tokens[row, :length] = torch.arange(1, length + 1)
        mask[row, 2:length] = 1
    return {}, tokens, mask, label


def test_batched_scores_match_single_examples():
    model = LookupModel()
    examples = [rendered([4, 5, 6, 4], 1), rendered([7, 4, 5, 6], 2)]
    batch_tokens, batch_mask, _ = collate_examples(examples)
    batch_sum, batch_average = score_batch(model, batch_tokens, batch_mask)
    for index, (_, tokens, mask, _) in enumerate(examples):
        logits, _ = model(tokens)
        expected_sum, expected_average = completion_losses(logits, tokens, mask)
        torch.testing.assert_close(batch_sum[index], expected_sum)
        torch.testing.assert_close(batch_average[index], expected_average)


def test_completion_mask_is_shifted_with_targets():
    tokens = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1]])
    logits = torch.zeros(1, 4, 4)
    logits[0, 0, 1] = 20
    logits[0, 1, 2] = 20
    logits[0, 2, 0] = 20
    sum_loss, average_loss = completion_losses(logits, tokens, mask)
    expected = torch.nn.functional.cross_entropy(
        logits[0, 1:3], tokens[0, 2:4], reduction="sum"
    )
    torch.testing.assert_close(sum_loss[0], expected)
    torch.testing.assert_close(average_loss[0], expected / 2)
    assert sum_loss.item() > 10


def test_wilson_interval_known_value_and_empty_input():
    low, high = wilson_interval(50, 100)
    assert abs(low - 0.4038) < 0.001
    assert abs(high - 0.5962) < 0.001
    assert wilson_interval(0, 0) is None


def test_routing_validity_mask_excludes_dynamic_padding():
    data = {"ctx_tokens": [1, 2], "ending_tokens": [[3], [4, 5], [6], [7, 8, 9]]}
    valid = _valid_choice_mask(data, 5)
    assert valid.sum(dim=1).tolist() == [3, 4, 3, 5]


def test_synthetic_fixture_verifies_acc_and_acc_norm():
    tokens = torch.ones(8, 4, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    logits = torch.zeros(8, 4, 4)
    lengths = [1, 2, 2, 2, 1, 2, 2, 2]
    target_strengths = [0.0, 0.5, -2.0, -2.0, 0.0, 3.0, -2.0, -2.0]
    for row, (length, strength) in enumerate(zip(lengths, target_strengths)):
        mask[row, 1 : 1 + length] = 1
        logits[row, :length, 1] = strength
    sum_loss, average_loss = completion_losses(logits, tokens, mask)
    raw_predictions = sum_loss.reshape(2, 4).argmin(dim=1)
    norm_predictions = average_loss.reshape(2, 4).argmin(dim=1)
    labels = torch.tensor([0, 1])
    assert (raw_predictions == labels).float().mean().item() == 1.0
    assert (norm_predictions == labels).float().mean().item() == 0.5
