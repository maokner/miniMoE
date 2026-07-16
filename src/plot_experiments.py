"""Render the committed portfolio figures from completed experiment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from matplotlib.colors import BoundaryNorm, ListedColormap


COLORS = {"base": "#2563eb", "sft": "#f97316"}


def load(path):
    return json.loads(Path(path).read_text())


def finish(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_hellaswag(base, sft, output):
    values = [base["metrics"]["acc_norm"], sft["metrics"]["acc_norm"]]
    intervals = [
        base["metrics"]["acc_norm_wilson_95"],
        sft["metrics"]["acc_norm_wilson_95"],
    ]
    errors = [
        [value - interval[0] for value, interval in zip(values, intervals)],
        [interval[1] - value for value, interval in zip(values, intervals)],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.bar(["Base", "SFT"], values, color=[COLORS["base"], COLORS["sft"]], width=0.58)
    ax.errorbar([0, 1], values, yerr=errors, fmt="none", ecolor="#111827", capsize=5)
    for index, value in enumerate(values):
        ax.text(
            index, value + max(values) * 0.025, f"{value:.1%}", ha="center", va="bottom"
        )
    ax.set_ylabel("HellaSwag normalized accuracy")
    ax.set_ylim(0, max(interval[1] for interval in intervals) * 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, output)


def plot_utilization(base, sft, output):
    matrices = [
        np.array([layer["utilization"] for layer in value["hellaswag"]["layers"]])
        for value in (base, sft)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, matrix, title in zip(axes, matrices, ("Base", "SFT")):
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=max(m.max() for m in matrices),
        )
        ax.set_title(title)
        ax.set_xlabel("Expert")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
    axes[0].set_ylabel("Layer")
    fig.colorbar(image, ax=axes, label="Share of selected routes", shrink=0.85)
    finish(fig, output)


def plot_router_stats(base, sft, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for name, value in (("base", base), ("sft", sft)):
        layers = value["hellaswag"]["layers"]
        x = [layer["layer"] for layer in layers]
        axes[0].plot(
            x,
            [layer["mean_router_entropy"] for layer in layers],
            marker="o",
            label=name.upper(),
            color=COLORS[name],
        )
        axes[1].plot(
            x,
            [layer["load_balance_cv"] for layer in layers],
            marker="o",
            label=name.upper(),
            color=COLORS[name],
        )
    axes[0].set_title("Router entropy")
    axes[0].set_ylabel("Mean entropy (nats)")
    axes[1].set_title("Load balance")
    axes[1].set_ylabel("Utilization coefficient of variation")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    finish(fig, output)


def plot_token_routes(base, sft, output):
    selected_id = "concise_explanation"
    records = []
    for name, value in (("Base", base), ("SFT", sft)):
        prompt = next(
            item
            for item in value["prompts"]["token_routes"]
            if item["prompt_id"] == selected_id
        )
        final_layer = max(route["layer"] for route in prompt["routes"])
        routes = [route for route in prompt["routes"] if route["layer"] == final_layer]
        records.append((name, routes))
    width = max(len(routes) for _, routes in records)
    matrix = np.full((2, width), np.nan)
    for row, (_, routes) in enumerate(records):
        matrix[row, : len(routes)] = [route["experts"][0] for route in routes]
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = next(
        item
        for item in base["prompts"]["token_routes"]
        if item["prompt_id"] == selected_id
    )["token_ids"]
    token_labels = []
    for token_id in token_ids:
        if token_id == 50256:
            token_labels.append("BOS")
            continue
        label = tokenizer.decode([token_id]).replace("\n", "↵").replace(" ", "·")
        token_labels.append(label or "∅")
    colors = ListedColormap(plt.get_cmap("tab10").colors[:8])
    norm = BoundaryNorm(np.arange(-0.5, 8.5), colors.N)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    image = ax.imshow(matrix, aspect="auto", cmap=colors, norm=norm)
    ax.set_yticks([0, 1], [name for name, _ in records])
    ax.set_xticks(range(len(token_labels)), token_labels, rotation=55, ha="right")
    ax.set_xlabel("Prompt token")
    ax.set_title("Top routed expert in final layer: concise explanation prompt")
    fig.colorbar(image, ax=ax, label="Top expert", ticks=range(8), shrink=0.9)
    finish(fig, output)


def plot_ablations(value, output):
    records = value["ablations"]
    label_map = {
        "learned_top2": "Learned top-2",
        "learned_top1": "Learned top-1",
        "random_top2": "Random top-2",
        "uniform_all": "Uniform all",
    }
    labels = [
        label_map.get(
            record["mode"], record["mode"].replace("ablate_expert_", "Remove E")
        )
        for record in records
    ]
    scores = [record["metrics"]["acc_norm"] for record in records]
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(range(len(scores)), scores, color="#4f46e5")
    for index, score in enumerate(scores):
        ax.text(
            index,
            score + max(scores) * 0.012,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, max(scores) * 1.16 if max(scores) else 1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("HellaSwag normalized accuracy")
    ax.set_title("Base checkpoint routing interventions (first 1,000 examples)")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()
    results = Path(args.results_dir)
    figures = Path(args.figures_dir)
    base_hs = load(results / "hellaswag_base.json")
    sft_hs = load(results / "hellaswag_sft.json")
    base_routing = load(results / "routing_base.json")
    sft_routing = load(results / "routing_sft.json")
    ablations = load(results / "ablations.json")
    plot_hellaswag(base_hs, sft_hs, figures / "hellaswag_base_vs_sft.png")
    plot_utilization(base_routing, sft_routing, figures / "expert_utilization.png")
    plot_router_stats(
        base_routing, sft_routing, figures / "router_entropy_load_balance.png"
    )
    plot_token_routes(base_routing, sft_routing, figures / "token_routing.png")
    plot_ablations(ablations, figures / "routing_ablations.png")


if __name__ == "__main__":
    main()
