"""So sanh nhieu model tu cac file metrics_*.json (do CAKD/eval_metrics.py sinh ra).

In bang so sanh + ve bieu do cot: Accuracy & macro-F1 moi model, va F1 tung lop.

Dung:
    python tools/compare_metrics.py \
      --metrics results/metrics_baseline_mobilenetv3_small.json \
                results/metrics_teacher.json \
                results/metrics_student_mobilenetv3_small.json \
      --out results/compare.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#4E79A7"
GREEN = "#59A14F"
GOLD = "#F28E2B"
RED = "#E15759"
GRID = "#B0B0B0"
BARS = [BLUE, GOLD, GREEN, RED, "#76B7B2", "#B07AA1"]

# Ten dep cho tung (model, arch)
_NICE = {
    ("baseline", "mobilenetv3_small"): "MobileNet (baseline)",
    ("student", "mobilenetv3_small"): "MobileNet (CAKD)",
    ("baseline", "resnet18"): "ResNet18 (baseline)",
    ("student", "resnet18"): "ResNet18 (CAKD)",
    ("teacher", "vit_b_16"): "ViT (teacher)",
}


def _label(d, fallback):
    return _NICE.get((d.get("model"), d.get("arch")), fallback)


def _style(ax, title, ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", color=GRID, alpha=0.3, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    p = argparse.ArgumentParser(description="So sánh nhiều model từ metrics_*.json")
    p.add_argument("--metrics", nargs="+", required=True, help="danh sách file metrics_*.json")
    p.add_argument("--out", default="compare.png", help="file PNG đầu ra")
    args = p.parse_args()

    data, labels = [], []
    for path in args.metrics:
        if not os.path.exists(path):
            print(f"⚠ bỏ qua (không thấy): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        data.append(d)
        labels.append(_label(d, os.path.splitext(os.path.basename(path))[0]))
    if not data:
        raise SystemExit("Không đọc được file metrics nào.")

    classes = data[0]["classes"]

    # ---- Bảng so sánh in ra màn hình ----
    print(f"\n{'model':<24}{'acc':>8}{'macro-P':>9}{'macro-R':>9}{'macro-F1':>10}")
    print("-" * 60)
    for lab, d in zip(labels, data):
        m = d["macro_avg"]
        print(f"{lab:<24}{d['accuracy']*100:>7.2f}%{m['precision']:>9.4f}"
              f"{m['recall']:>9.4f}{m['f1-score']:>10.4f}")
    print("\nF1 từng lớp:")
    print(f"{'model':<24}" + "".join(f"{c[:8]:>10}" for c in classes))
    for lab, d in zip(labels, data):
        print(f"{lab:<24}" + "".join(f"{d['per_class'][c]['f1-score']:>10.4f}" for c in classes))

    # ---- Biểu đồ ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    n = len(data)

    # Panel 1: Accuracy & macro-F1 mỗi model
    x = range(n)
    acc = [d["accuracy"] * 100 for d in data]
    f1 = [d["macro_avg"]["f1-score"] * 100 for d in data]
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], acc, width=w, color=BLUE, label="Accuracy")
    axes[0].bar([i + w / 2 for i in x], f1, width=w, color=GOLD, label="macro-F1")
    for i in x:
        axes[0].text(i - w / 2, acc[i] + 0.5, f"{acc[i]:.1f}", ha="center", fontsize=8, color=BLUE)
        axes[0].text(i + w / 2, f1[i] + 0.5, f"{f1[i]:.1f}", ha="center", fontsize=8, color=GOLD)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylim(0, 105)
    _style(axes[0], "Accuracy & macro-F1 (%)", ylabel="%")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    # Panel 2: F1 từng lớp, nhóm theo model
    nc = len(classes)
    gw = 0.8 / n
    for j, (lab, d) in enumerate(zip(labels, data)):
        vals = [d["per_class"][c]["f1-score"] * 100 for c in classes]
        xs = [k - 0.4 + gw * (j + 0.5) for k in range(nc)]
        axes[1].bar(xs, vals, width=gw, color=BARS[j % len(BARS)], label=lab)
    axes[1].set_xticks(range(nc))
    axes[1].set_xticklabels(classes, fontsize=9)
    axes[1].set_ylim(0, 105)
    _style(axes[1], "F1 từng lớp (%)", ylabel="F1 (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("So sánh model — baseline vs teacher vs CAKD", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nĐã lưu biểu đồ so sánh: {args.out}")


if __name__ == "__main__":
    main()
