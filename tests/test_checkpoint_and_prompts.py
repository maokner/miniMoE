import copy

import pytest
import torch

from checkpoint import load_checkpoint, sha256_file
from experiments import greedy_generate
from model import Model, ModelConfig
from prompts import load_manifest, validate_manifest


def tiny_config():
    return ModelConfig(
        max_seq_length=12,
        vocab_size=64,
        num_layers=1,
        hidden_dim=8,
        num_experts=8,
        top_k=2,
    )


def test_checkpoint_load_preserves_logits_and_greedy_choice(tmp_path):
    torch.manual_seed(67)
    config = tiny_config()
    original = Model(config).eval()
    path = tmp_path / "tiny.pt"
    torch.save(
        {
            "model": original.state_dict(),
            "model_config": vars(config),
            "step": 19,
            "tokens_seen": 1234,
        },
        path,
    )
    loaded, loaded_config, metadata = load_checkpoint(path)
    tokens = torch.tensor([[1, 2, 3]])
    original_logits, _ = original(tokens)
    loaded_logits, _ = loaded(tokens)
    torch.testing.assert_close(original_logits, loaded_logits)
    assert (
        original_logits[:, -1].argmax().item() == loaded_logits[:, -1].argmax().item()
    )
    assert loaded_config == config
    assert metadata["sha256"] == sha256_file(path)
    assert metadata["step"] == 19


def test_prompt_manifest_is_complete_and_rejects_duplicates():
    manifest = load_manifest()
    assert len(manifest["prompts"]) == 12
    invalid = copy.deepcopy(manifest)
    invalid["prompts"][1]["id"] = invalid["prompts"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(invalid)


def test_greedy_generation_is_deterministic():
    torch.manual_seed(67)
    config = tiny_config()
    model = Model(config).eval()
    first = greedy_generate(model, [1, 2, 3], config.max_seq_length, max_new_tokens=5)
    torch.manual_seed(999)
    second = greedy_generate(model, [1, 2, 3], config.max_seq_length, max_new_tokens=5)
    assert first == second
