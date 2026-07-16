"""Unified, resumable experiment CLI for the final miniMoE checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tiktoken
import torch

from checkpoint import load_checkpoint
from eval_utils import (
    append_jsonl,
    autocast_context,
    barrier,
    close_distributed,
    completed_indices,
    distributed_context,
    merge_rank_parts,
    part_path,
    result_metadata,
    seed_everything,
    write_json,
)
from hellaswag import (
    collate_examples,
    dataset_path,
    download,
    file_sha256,
    hellaswags,
    iterate_examples,
    render_example,
    score_batch,
    wilson_interval,
)
from model import EOS_TOKEN_ID
from prompts import load_manifest, render_prompt
from routing import RoutingCollector, merge_routing_summaries


DEFAULT_BASE = "checkpoints/minimoe_step_0019073.pt"
DEFAULT_SFT = "checkpoints/minimoe_sft.pt"
DEFAULT_RESULTS = Path("results")


def ensure_dataset(context, split="val"):
    """Download once on rank zero before any distributed readers proceed."""
    if context.is_primary:
        download(split)
    barrier(context)


def dataset_metadata(split="val"):
    path = dataset_path(split)
    return {
        "name": "HellaSwag",
        "split": split,
        "url": hellaswags[split],
        "path": path,
        "sha256": file_sha256(path),
    }


def batched(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _prediction_metrics(predictions):
    valid = [record for record in predictions if not record.get("skipped")]
    skipped = len(predictions) - len(valid)
    correct = sum(record["prediction"] == record["label"] for record in valid)
    correct_norm = sum(record["prediction_norm"] == record["label"] for record in valid)
    total = len(valid)
    return {
        "acc": correct / total if total else None,
        "acc_norm": correct_norm / total if total else None,
        "acc_wilson_95": wilson_interval(correct, total),
        "acc_norm_wilson_95": wilson_interval(correct_norm, total),
        "correct": correct,
        "correct_norm": correct_norm,
        "total": total,
        "skipped": skipped,
    }


def evaluate_hellaswag(
    checkpoint_path,
    output,
    context,
    batch_size=8,
    max_examples=0,
    seed=67,
    routing_mode="learned_top2",
    tag=None,
    compile_model=False,
    autocast=True,
    loaded=None,
):
    ensure_dataset(context)
    seed_everything(seed, context.rank)
    if loaded is None:
        model, config, checkpoint = load_checkpoint(checkpoint_path, context.device)
    else:
        model, config, checkpoint = loaded
    model.set_routing(routing_mode)
    if compile_model:
        model = torch.compile(model)

    output = Path(output)
    part = part_path(output, context.rank, context.world_size, tag)
    done = completed_indices(part)
    pending = []
    for index, example in enumerate(iterate_examples("val")):
        if max_examples and index >= max_examples:
            break
        if index % context.world_size == context.rank and index not in done:
            pending.append((index, example))

    started = time.perf_counter()
    model.eval()
    evaluation_batch_size = 1 if routing_mode == "random_top2" else batch_size
    with torch.no_grad():
        for batch in batched(pending, evaluation_batch_size):
            rendered = []
            indexes = []
            skipped = []
            for index, example in batch:
                item = render_example(example)
                if item[1].size(1) > config.max_seq_length:
                    skipped.append(
                        {
                            "index": index,
                            "label": int(item[3]),
                            "skipped": True,
                            "reason": "context_length",
                            "sequence_length": item[1].size(1),
                        }
                    )
                else:
                    indexes.append(index)
                    rendered.append(item)
            if skipped:
                append_jsonl(part, skipped)
            if not rendered:
                continue

            tokens, mask, labels = collate_examples(rendered)
            tokens = tokens.to(context.device, non_blocking=True)
            mask = mask.to(context.device, non_blocking=True)
            if routing_mode == "random_top2":
                torch.manual_seed(seed + indexes[0])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + indexes[0])
            with autocast_context(context.device, autocast):
                sum_losses, avg_losses = score_batch(model, tokens, mask)
            records = []
            for row, index in enumerate(indexes):
                records.append(
                    {
                        "index": index,
                        "label": int(labels[row]),
                        "prediction": int(sum_losses[row].argmin()),
                        "prediction_norm": int(avg_losses[row].argmin()),
                        "sum_losses": sum_losses[row].float().cpu().tolist(),
                        "avg_losses": avg_losses[row].float().cpu().tolist(),
                        "sequence_length": int(rendered[row][1].size(1)),
                        "skipped": False,
                    }
                )
            append_jsonl(part, records)

    elapsed = time.perf_counter() - started
    barrier(context)
    result = None
    if context.is_primary:
        parts = [
            part_path(output, rank, context.world_size, tag)
            for rank in range(context.world_size)
        ]
        expected = max_examples or 10042
        # A resumed run with a smaller --max-examples may find extra completed
        # indices in its parts; they are valid records, just out of scope.
        predictions = [
            record
            for record in merge_rank_parts(parts)
            if int(record["index"]) < expected
        ]
        if len(predictions) != expected:
            raise RuntimeError(
                f"Expected {expected} results, found {len(predictions)}; rerun to resume"
            )
        settings = {
            "batch_size_per_gpu": batch_size,
            "max_examples": max_examples or None,
            "seed": seed,
            "routing_mode": routing_mode,
            "autocast": "cuda_bfloat16" if autocast else "disabled",
            "compile": compile_model,
        }
        result = result_metadata(
            "hellaswag",
            context,
            settings,
            checkpoints=[checkpoint],
            dataset=dataset_metadata(),
        )
        result.update(
            {
                "metrics": _prediction_metrics(predictions),
                "timing": {
                    "rank_zero_elapsed_seconds_this_invocation": elapsed,
                    "examples_per_second": len(predictions) / elapsed
                    if elapsed
                    else None,
                },
                "predictions": predictions,
            }
        )
        write_json(output, result)
    barrier(context)
    return result


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_seq_length, max_new_tokens=96):
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new_tokens):
        with autocast_context(str(device)):
            logits, _ = model(tokens[:, -max_seq_length:])
        next_token = int(logits[:, -1].argmax(dim=-1).item())
        if next_token == EOS_TOKEN_ID:
            break
        generated.append(next_token)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )
    return generated


def run_prompts(args, context):
    if not context.is_primary:
        barrier(context)
        return None
    tokenizer = tiktoken.get_encoding("gpt2")
    manifest = load_manifest(args.manifest)
    checkpoints = []
    outputs = []
    for checkpoint_name, checkpoint_path in (("base", args.base), ("sft", args.sft)):
        seed_everything(args.seed)
        model, config, metadata = load_checkpoint(checkpoint_path, context.device)
        checkpoints.append(metadata)
        for prompt in manifest["prompts"]:
            if prompt["mode"] == "raw_base_only" and checkpoint_name != "base":
                continue
            prompt_ids = render_prompt(prompt, tokenizer)
            generated_ids = greedy_generate(
                model,
                prompt_ids[-config.max_seq_length :],
                config.max_seq_length,
                args.max_new_tokens,
            )
            outputs.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "mode": prompt["mode"],
                    "checkpoint": checkpoint_name,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated_ids,
                    "output": tokenizer.decode(generated_ids),
                }
            )
        del model
    result = result_metadata(
        "prompts",
        context,
        {"seed": args.seed, "temperature": 0, "max_new_tokens": args.max_new_tokens},
        checkpoints=checkpoints,
    )
    result.update({"manifest": manifest, "outputs": outputs})
    write_json(args.output, result)
    barrier(context)
    return result


def _valid_choice_mask(data, max_len):
    rows = []
    for ending in data["ending_tokens"]:
        length = len(data["ctx_tokens"]) + len(ending)
        rows.append([True] * length + [False] * (max_len - length))
    return torch.tensor(rows, dtype=torch.bool)


def _collect_routing_for_hellaswag(
    model, config, context, tokenizer, max_examples, batch_size
):
    collector = RoutingCollector(config.num_layers, config.num_experts)
    model.set_routing("learned_top2", collector)
    owned = []
    for index, example in enumerate(iterate_examples("val")):
        if index >= max_examples:
            break
        if index % context.world_size == context.rank:
            owned.append((index, example))
    with torch.no_grad():
        for batch in batched(owned, batch_size):
            rendered = [render_example(example, tokenizer) for _, example in batch]
            rendered = [
                item for item in rendered if item[1].size(1) <= config.max_seq_length
            ]
            if not rendered:
                continue
            tokens, _, _ = collate_examples(rendered)
            valid = torch.stack(
                [_valid_choice_mask(item[0], tokens.size(-1)) for item in rendered]
            )
            flat_tokens = tokens.flatten(0, 1).to(context.device)
            collector.expect_mask(valid.flatten(0, 1))
            with autocast_context(context.device):
                model(flat_tokens)
    return collector.summary()


def _collect_routing_for_prompts(model, config, context, tokenizer, manifest):
    collector = RoutingCollector(
        config.num_layers, config.num_experts, capture_tokens=True
    )
    model.set_routing("learned_top2", collector)
    prompt_records = []
    with torch.no_grad():
        for prompt_index, prompt in enumerate(manifest["prompts"]):
            if prompt_index % context.world_size != context.rank:
                continue
            ids = render_prompt(prompt, tokenizer)[-config.max_seq_length :]
            tokens = torch.tensor([ids], dtype=torch.long, device=context.device)
            start = len(collector.token_routes)
            collector.expect_mask(torch.ones_like(tokens, dtype=torch.bool))
            with autocast_context(context.device):
                model(tokens)
            prompt_records.append(
                {
                    "prompt_id": prompt["id"],
                    "token_ids": ids,
                    "routes": collector.token_routes[start:],
                }
            )
    summary = collector.summary()
    summary["token_routes"] = prompt_records
    return summary


def run_routing(args, context):
    ensure_dataset(context)
    tokenizer = tiktoken.get_encoding("gpt2")
    manifest = load_manifest(args.manifest)
    seed_everything(args.seed, context.rank)
    model, config, checkpoint = load_checkpoint(args.checkpoint, context.device)
    hellaswag_summary = _collect_routing_for_hellaswag(
        model, config, context, tokenizer, args.max_examples, args.batch_size
    )
    prompt_summary = _collect_routing_for_prompts(
        model, config, context, tokenizer, manifest
    )
    part = part_path(args.output, context.rank, context.world_size)
    write_json(
        part.with_suffix(".json"),
        {"hellaswag": hellaswag_summary, "prompts": prompt_summary},
    )
    barrier(context)
    if context.is_primary:
        rank_values = [
            json.loads(
                part_path(args.output, rank, context.world_size)
                .with_suffix(".json")
                .read_text()
            )
            for rank in range(context.world_size)
        ]
        result = result_metadata(
            "routing",
            context,
            {
                "seed": args.seed,
                "hellaswag_examples": args.max_examples,
                "batch_size_per_gpu": args.batch_size,
            },
            checkpoints=[checkpoint],
            dataset=dataset_metadata(),
        )
        result["warning"] = (
            "Routing differences are descriptive and do not by themselves establish semantic expert specialization."
        )
        result["hellaswag"] = merge_routing_summaries(
            [value["hellaswag"] for value in rank_values]
        )
        prompt_summaries = [value["prompts"] for value in rank_values]
        result["prompts"] = merge_routing_summaries(prompt_summaries)
        result["prompts"]["token_routes"] = sorted(
            [
                record
                for summary in prompt_summaries
                for record in summary["token_routes"]
            ],
            key=lambda record: record["prompt_id"],
        )
        write_json(args.output, result)
        return result
    return None


def run_ablations(args, context):
    modes = ["learned_top2", "learned_top1", "random_top2", "uniform_all"]
    modes.extend(f"ablate_expert_{index}" for index in range(8))
    summaries = []
    loaded = load_checkpoint(args.checkpoint, context.device)
    for mode in modes:
        mode_output = Path(args.output).parent / ".parts" / f"ablation-{mode}.json"
        result = evaluate_hellaswag(
            args.checkpoint,
            mode_output,
            context,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
            seed=args.seed,
            routing_mode=mode,
            tag=mode,
            loaded=loaded,
        )
        if context.is_primary:
            summaries.append(
                {"mode": mode, "metrics": result["metrics"], "timing": result["timing"]}
            )
    if context.is_primary:
        checkpoint = loaded[2]
        result = result_metadata(
            "ablations",
            context,
            {
                "seed": args.seed,
                "max_examples": args.max_examples,
                "batch_size_per_gpu": args.batch_size,
            },
            checkpoints=[checkpoint],
            dataset=dataset_metadata(),
        )
        result["ablations"] = summaries
        write_json(args.output, result)
        return result
    return None


def run_smoke(args, context):
    results = {}
    for name, checkpoint in (("base", args.base), ("sft", args.sft)):
        output = Path(args.output).parent / ".parts" / f"smoke-{name}.json"
        results[name] = evaluate_hellaswag(
            checkpoint,
            output,
            context,
            batch_size=args.batch_size,
            max_examples=32,
            seed=args.seed,
            tag=name,
        )
    if context.is_primary:
        tokenizer = tiktoken.get_encoding("gpt2")
        prompt = load_manifest(args.manifest)["prompts"][2]
        generations = {}
        routing = {}
        checkpoints = []
        for name, checkpoint_path in (("base", args.base), ("sft", args.sft)):
            model, config, metadata = load_checkpoint(checkpoint_path, context.device)
            checkpoints.append(metadata)
            ids = render_prompt(prompt, tokenizer)[-config.max_seq_length :]
            collector = RoutingCollector(config.num_layers, config.num_experts)
            model.set_routing("learned_top2", collector)
            tokens = torch.tensor([ids], dtype=torch.long, device=context.device)
            collector.expect_mask(torch.ones_like(tokens, dtype=torch.bool))
            with torch.no_grad(), autocast_context(context.device):
                logits, _ = model(tokens)
            if logits.shape[:2] != tokens.shape:
                raise RuntimeError("Smoke forward pass returned an unexpected shape")
            routing[name] = collector.summary()
            model.set_routing("learned_top2")
            generated = greedy_generate(model, ids, config.max_seq_length, 16)
            generations[name] = {
                "token_ids": generated,
                "text": tokenizer.decode(generated),
            }
        result = result_metadata(
            "smoke",
            context,
            {
                "seed": args.seed,
                "hellaswag_examples": 32,
                "batch_size_per_gpu": args.batch_size,
            },
            checkpoints=checkpoints,
            dataset=dataset_metadata(),
        )
        result.update(
            {
                "hellaswag": {
                    name: value["metrics"] for name, value in results.items()
                },
                "predictions": {
                    name: value["predictions"] for name, value in results.items()
                },
                "routing": routing,
                "generations": generations,
            }
        )
        write_json(args.output, result)
        return result
    return None


def add_common(parser):
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--batch-size", type=int, default=8, help="Examples per GPU")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    add_common(smoke)
    smoke.add_argument("--base", default=DEFAULT_BASE)
    smoke.add_argument("--sft", default=DEFAULT_SFT)
    smoke.add_argument("--manifest", default=None)
    smoke.add_argument("--output", default=str(DEFAULT_RESULTS / "smoke.json"))

    hellaswag = subparsers.add_parser("hellaswag")
    add_common(hellaswag)
    hellaswag.add_argument("--checkpoint", required=True)
    hellaswag.add_argument("--output", required=True)
    hellaswag.add_argument("--max-examples", type=int, default=0)
    hellaswag.add_argument("--compile", action="store_true")
    hellaswag.add_argument("--no-autocast", action="store_true")

    routing = subparsers.add_parser("routing")
    add_common(routing)
    routing.add_argument("--checkpoint", required=True)
    routing.add_argument("--output", required=True)
    routing.add_argument("--max-examples", type=int, default=1000)
    routing.add_argument("--manifest", default=None)

    ablations = subparsers.add_parser("ablations")
    add_common(ablations)
    ablations.add_argument("--checkpoint", default=DEFAULT_BASE)
    ablations.add_argument("--output", default=str(DEFAULT_RESULTS / "ablations.json"))
    ablations.add_argument("--max-examples", type=int, default=1000)

    prompts = subparsers.add_parser("prompts")
    add_common(prompts)
    prompts.add_argument("--base", default=DEFAULT_BASE)
    prompts.add_argument("--sft", default=DEFAULT_SFT)
    prompts.add_argument("--manifest", default=None)
    prompts.add_argument("--max-new-tokens", type=int, default=96)
    prompts.add_argument(
        "--output", default=str(DEFAULT_RESULTS / "prompt_outputs.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    context = distributed_context(args.device)
    try:
        if args.command == "smoke":
            run_smoke(args, context)
        elif args.command == "hellaswag":
            evaluate_hellaswag(
                args.checkpoint,
                args.output,
                context,
                batch_size=args.batch_size,
                max_examples=args.max_examples,
                seed=args.seed,
                compile_model=args.compile,
                autocast=not args.no_autocast,
            )
        elif args.command == "routing":
            run_routing(args, context)
        elif args.command == "ablations":
            run_ablations(args, context)
        elif args.command == "prompts":
            run_prompts(args, context)
    finally:
        close_distributed(context)


if __name__ == "__main__":
    main()
