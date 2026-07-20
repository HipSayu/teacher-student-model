"""Ve bieu do qua trinh train tu file history JSON (do dist_train_*.py sinh ra).

Dung:
    python tools/plot_training.py --history /kaggle/working/history_teacher.json
    python tools/plot_training.py --history /kaggle/working/history_cakd.json --out /kaggle/working/plot_cakd.png
    python tools/plot_training.py --history history_cakd.json --title "CAKD student"

History la list cac dict {epoch, train_loss, train_acc1, test_acc1, [cls_loss, pca_loss, gl_loss, gan_loss], lr}.
Tu dong nhan dien teacher (khong co loss thanh phan) hay cakd (co).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # backend khong can man hinh (chay tren Kaggle/headless)
import matplotlib.pyplot as plt

# Bang mau phan biet duoc cho nguoi mu mau, gan co dinh theo tung duong:
BLUE = "#4E79A7"    # cls / train acc / train loss
GREEN = "#59A14F"   # pca / learning rate
GOLD = "#F28E2B"    # gl
PURPLE = "#6A4C93"  # gan
RED = "#E15759"     # val acc
GRID = "#B0B0B0"


def _style(ax, title, xlabel="epoch", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=GRID, alpha=0.3, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _line(ax, xs, ys, color, label, end_label=False):
    """Ve 1 duong. Nhieu epoch (>40) thi bo marker cho sach. end_label=ghi ten o cuoi duong."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    xs2, ys2 = zip(*pts)
    ms = 3.5 if len(xs2) <= 40 else 0
    ax.plot(xs2, ys2, color=color, linewidth=1.8,
            marker="o" if ms else "", markersize=ms, label=label)
    if end_label:
        ax.annotate(label, xy=(xs2[-1], ys2[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.5, fontweight="bold", color=color)


def _mark_best(ax, xs, ys, color):
    """Cham tron tai dinh val acc + chu den dam 'best XX.XX% @epYY' o goc tren-trai."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    bx, by = max(pts, key=lambda p: p[1])
    ax.scatter([bx], [by], s=55, marker="o", color=color, zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.text(0.03, 0.94, f"best {by:.2f}% @ep{bx}", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#222222", va="top", ha="left")


def _plot_lr(ax, xs, lrs):
    """Panel learning rate: thang LOG de thay ro cac bac giam (warmup + step/cosine)."""
    pts = [(x, y) for x, y in zip(xs, lrs) if y is not None and y > 0]
    if not pts:
        _style(ax, "Learning rate", ylabel="lr")
        return
    xs2, ys2 = zip(*pts)
    ms = 3.5 if len(xs2) <= 40 else 0
    ax.plot(xs2, ys2, color=GREEN, linewidth=1.8,
            marker="o" if ms else "", markersize=ms, label="lr")
    ax.set_yscale("log")
    _style(ax, "Learning rate", ylabel="lr")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot(history, out_path, title=None):
    ep = [r["epoch"] for r in history]
    is_cakd = any(r.get("gl_loss") is not None for r in history)
    has_lr = any(r.get("lr") is not None for r in history)
    tr = [r.get("train_acc1") for r in history]
    va = [r.get("test_acc1") for r in history]
    lr = [r.get("lr") for r in history]

    if title is None:
        title = "CAKD student" if is_cakd else "Teacher"
    suptitle = f"{title} — {len(history)} epoch"

    if is_cakd:
        ncol = 4 if has_lr else 3
        fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 4.3))
        # Panel 1: loss tong (train)
        _line(axes[0], ep, [r.get("train_loss") for r in history], BLUE, "train loss")
        _style(axes[0], "Loss tổng (train)", ylabel="loss")
        axes[0].legend(frameon=False, fontsize=8)
        # Panel 2: cac loss thanh phan (log scale) + nhan cuoi duong + legend 2 cot
        _line(axes[1], ep, [r.get("cls_loss") for r in history], BLUE, "cls", end_label=True)
        _line(axes[1], ep, [r.get("pca_loss") for r in history], GREEN, "pca", end_label=True)
        _line(axes[1], ep, [r.get("gl_loss") for r in history], GOLD, "gl", end_label=True)
        _line(axes[1], ep, [r.get("gan_loss") for r in history], PURPLE, "gan", end_label=True)
        axes[1].set_yscale("log")
        _style(axes[1], "Các loss thành phần (log)", ylabel="loss")
        axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="center")
        # Panel 3: accuracy
        _line(axes[2], ep, tr, BLUE, "train acc@1")
        _line(axes[2], ep, va, RED, "val acc@1")
        _mark_best(axes[2], ep, va, RED)
        axes[2].margins(y=0.10)
        _style(axes[2], "Accuracy", ylabel="acc@1 (%)")
        axes[2].legend(frameon=False, fontsize=8, loc="lower right")
        # Panel 4: learning rate (neu co)
        if has_lr:
            _plot_lr(axes[3], ep, lr)
    else:
        ncol = 3 if has_lr else 2
        fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 4.3))
        _line(axes[0], ep, [r.get("train_loss") for r in history], BLUE, "train loss")
        _style(axes[0], "Loss (train)", ylabel="loss")
        axes[0].legend(frameon=False, fontsize=8)
        _line(axes[1], ep, tr, BLUE, "train acc@1")
        _line(axes[1], ep, va, RED, "val acc@1")
        _mark_best(axes[1], ep, va, RED)
        axes[1].margins(y=0.10)
        _style(axes[1], "Accuracy", ylabel="acc@1 (%)")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")
        if has_lr:
            _plot_lr(axes[2], ep, lr)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {out_path}")

    # In tom tat dong cuoi
    last = history[-1]
    print(f"Epoch cuối ({last['epoch']}): "
          f"train_acc1={last.get('train_acc1')}  val_acc1={last.get('test_acc1')}")
    best = max((r.get("test_acc1") or 0) for r in history)
    print(f"Val acc@1 tốt nhất: {best:.3f}")


def main():
    p = argparse.ArgumentParser(description="Vẽ biểu đồ quá trình train từ history JSON")
    p.add_argument("--history", required=True, help="đường dẫn file history_*.json")
    p.add_argument("--out", default=None, help="file PNG đầu ra (mặc định cạnh file history)")
    p.add_argument("--title", default=None,
                   help="tiêu đề (mặc định: 'CAKD student' hoặc 'Teacher')")
    args = p.parse_args()

    with open(args.history) as f:
        history = json.load(f)
    if not history:
        raise SystemExit("File history rỗng — chưa có epoch nào được ghi.")

    out = args.out or os.path.splitext(args.history)[0] + ".png"
    plot(history, out, title=args.title)


if __name__ == "__main__":
    main()
