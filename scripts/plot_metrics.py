import argparse
import csv
import os


GPT2_ACC_NORM = 0.2955
MINIMOE_TOTAL_PARAMS = 280.4
MINIMOE_ACTIVE_PARAMS = 110.4
GPT2_PARAMS = 124.4

COLORS = {
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "moe": "#2563eb",
    "gpt2": "#059669",
    "val": "#dc2626",
    "aux": "#f59e0b",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key]) if row.get(key) else None


def train_rows(rows):
    return [row for row in rows if row["phase"] == "train"]


def last_test_loss(rows):
    tests = [row for row in rows if row["phase"] == "test" and row.get("test_loss")]
    return f(tests[-1], "test_loss") if tests else None


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, value, size=13, color=None, weight=400, anchor="start"):
    color = color or COLORS["ink"]
    return (
        f'<text x="{x}" y="{y}" font-family="Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def start_svg(width=900, height=480):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def write_svg(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(items + ["</svg>"]))


def line(points, color, width=3, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'


def chart(title, subtitle, width=900, height=480):
    items = start_svg(width, height)
    items.append(text(28, 36, title, 22, weight=700))
    items.append(text(28, 60, subtitle, 13, COLORS["muted"]))
    return items


def scale(values, a, b):
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return lambda value: (a + b) / 2
    return lambda value: a + (value - lo) / (hi - lo) * (b - a)


def axes(items, x_values, y_values, y_label, y_format=lambda v: f"{v:.2f}"):
    left, top, right, bottom = 72, 82, 24, 62
    width, height = 900, 480
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_map = scale(x_values, left, left + plot_w)
    y_map_raw = scale(y_values, top + plot_h, top)
    items.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="{COLORS["grid"]}"/>')
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y_value = min(y_values) + (max(y_values) - min(y_values)) * frac
        y = y_map_raw(y_value)
        items.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        items.append(text(left - 10, y + 4, y_format(y_value), 12, COLORS["muted"], anchor="end"))
    for value in [0, 2, 4, 6, 8, 10]:
        if min(x_values) <= value <= max(x_values):
            x = x_map(value)
            items.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
            items.append(text(x, top + plot_h + 24, value, 12, COLORS["muted"], anchor="middle"))
    items.append(text(left + plot_w / 2, top + plot_h + 50, "tokens seen, billions", 12, COLORS["muted"], anchor="middle"))
    items.append(text(16, top + plot_h / 2, y_label, 12, COLORS["muted"]))
    return x_map, y_map_raw


def legend(items, entries, x=620, y=36):
    offset = 0
    for label, color in entries:
        items.append(f'<line x1="{x + offset}" y1="{y}" x2="{x + offset + 24}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        items.append(text(x + offset + 32, y + 4, label, 12, COLORS["muted"]))
        offset += 128


def plot_loss(rows, out_dir):
    all_rows = rows
    rows = train_rows(all_rows)
    x = [f(row, "tokens_seen") / 1e9 for row in rows]
    train = moving_average([f(row, "train_loss") for row in rows], 7)
    val_points = [(f(row, "tokens_seen") / 1e9, f(row, "val_loss")) for row in rows if f(row, "val_loss") is not None]
    val_x = [p[0] for p in val_points]
    val = moving_average([p[1] for p in val_points], 7)
    test_loss = last_test_loss(all_rows)
    y_values = train + val + ([test_loss] if test_loss else [])
    items = chart("Loss curves", "7-point moving average from the 10B-token training run")
    x_map, y_map = axes(items, x, y_values, "loss")
    items.append(line([(x_map(a), y_map(b)) for a, b in zip(x, train)], COLORS["moe"]))
    items.append(line([(x_map(a), y_map(b)) for a, b in zip(val_x, val)], COLORS["val"]))
    if test_loss:
        items.append(f'<circle cx="{x_map(max(x)):.1f}" cy="{y_map(test_loss):.1f}" r="5" fill="{COLORS["ink"]}"/>')
        items.append(text(x_map(max(x)) - 8, y_map(test_loss) - 12, f"test {test_loss:.3f}", 12, COLORS["ink"], anchor="end"))
    legend(items, [("train", COLORS["moe"]), ("validation", COLORS["val"])])
    write_svg(os.path.join(out_dir, "loss_curves.svg"), items)


def plot_hellaswag(rows, out_dir):
    rows = train_rows(rows)
    points = [(f(row, "tokens_seen") / 1e9, f(row, "hellaswag_acc_norm")) for row in rows if f(row, "hellaswag_acc_norm") is not None]
    x = [p[0] for p in points]
    acc_norm = moving_average([p[1] for p in points], 9)
    y_values = acc_norm + [0.25, GPT2_ACC_NORM]
    items = chart("HellaSwag normalized accuracy", "9-point moving average of 32-example training probes")
    x_map, y_map = axes(items, x, y_values, "accuracy", lambda v: f"{v * 100:.0f}%")
    items.append(line([(x_map(a), y_map(b)) for a, b in zip(x, acc_norm)], COLORS["moe"]))
    for label, value, color in [
        ("GPT-2 full val 29.55%", GPT2_ACC_NORM, COLORS["gpt2"]),
        ("random 25%", 0.25, COLORS["muted"]),
    ]:
        items.append(line([(x_map(min(x)), y_map(value)), (x_map(max(x)), y_map(value))], color, 2, "6 6"))
        items.append(text(x_map(max(x)) - 4, y_map(value) - 8, label, 12, color, anchor="end"))
    legend(items, [("miniMoE", COLORS["moe"])], 720)
    write_svg(os.path.join(out_dir, "hellaswag.svg"), items)


def plot_baseline(rows, out_dir):
    final = [row for row in train_rows(rows) if f(row, "hellaswag_acc_norm") is not None][-1]
    mini_acc = f(final, "hellaswag_acc_norm") * 100
    width = 1040
    height = 500
    items = start_svg(width, height)
    items.append(text(42, 48, "miniMoE vs GPT-2 small", 26, weight=700))
    items.append(text(42, 78, "Sparse capacity, active compute, and normalized HellaSwag accuracy", 15, COLORS["muted"]))
    rows_for_plot = [
        ("Total parameters", MINIMOE_TOTAL_PARAMS, GPT2_PARAMS, "M", 300),
        ("Active parameters per token", MINIMOE_ACTIVE_PARAMS, GPT2_PARAMS, "M", 140),
        ("HellaSwag normalized accuracy", mini_acc, GPT2_ACC_NORM * 100, "%", 40),
    ]
    label_x = 42
    model_x = 42
    bar_x = 150
    bar_max_width = 650
    for i, (label, mini, gpt2, unit, max_value) in enumerate(rows_for_plot):
        y = 132 + i * 112
        items.append(text(label_x, y, label, 17, weight=700))
        for j, (name, value, color) in enumerate([("miniMoE", mini, COLORS["moe"]), ("GPT-2", gpt2, COLORS["gpt2"])]):
            bar_y = y + 22 + j * 32
            bar_width = value / max_value * bar_max_width
            items.append(text(model_x, bar_y + 17, name, 14, COLORS["muted"]))
            items.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width:.1f}" height="22" rx="5" fill="{color}"/>')
            items.append(text(bar_x + bar_width + 14, bar_y + 17, f"{value:.1f}{unit}", 14))
    write_svg(os.path.join(out_dir, "gpt2_baseline.svg"), items)


def plot_optimization(rows, out_dir):
    rows = train_rows(rows)
    x = [f(row, "tokens_seen") / 1e9 for row in rows]
    grad = moving_average([f(row, "grad_norm") for row in rows], 7)
    lr = moving_average([f(row, "lr") * 1e4 for row in rows], 7)
    y_values = grad + lr
    items = chart("Optimization", "Gradient norm and learning rate schedule, smoothed over 7 points")
    x_map, y_map = axes(items, x, y_values, "grad norm / lr x 1e4")
    items.append(line([(x_map(a), y_map(b)) for a, b in zip(x, grad)], COLORS["aux"]))
    items.append(line([(x_map(a), y_map(b)) for a, b in zip(x, lr)], COLORS["moe"]))
    legend(items, [("grad norm", COLORS["aux"]), ("lr x 1e4", COLORS["moe"])])
    write_svg(os.path.join(out_dir, "optimization.svg"), items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="train_log.csv")
    parser.add_argument("--out-dir", default="assets")
    args = parser.parse_args()
    rows = read_csv(args.log)
    plot_loss(rows, args.out_dir)
    plot_hellaswag(rows, args.out_dir)
    plot_baseline(rows, args.out_dir)
    plot_optimization(rows, args.out_dir)


if __name__ == "__main__":
    main()
