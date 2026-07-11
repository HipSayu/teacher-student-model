"""Ve bieu do qua trinh train tu file history JSON (do dist_train_*.py sinh ra).

Dung:
    python tools/plot_training.py --history /kaggle/working/history_teacher.json
    python tools/plot_training.py --history /kaggle/working/history_cakd.json --out /kaggle/working/plot_cakd.png

History la list cac dict {epoch, train_loss, train_acc1, test_acc1, [cls_loss, pca_loss, gl_loss, gan_loss]}.
Tu dong nhan dien teacher (khong co loss thanh phan) hay cakd (co).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # backend khong can man hinh (chay tren Kaggle/headless)
import matplotlib.pyplot as plt

# Bang mau phan biet duoc cho nguoi mu mau (Tableau-10), gan theo tung duong:
BLUE = "#4E79A7"    # train
ORANGE = "#F28E2B"  # val
GREEN = "#59A14F"
PURPLE = "#B07AA1"
GRID = "#B0B0B0"


def _style(ax, title, xlabel="epoch", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=GRID, alpha=0.3, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _line(ax, xs, ys, color, label):
    ys2 = [y for y in ys if y is not None]
    xs2 = [x for x, y in zip(xs, ys) if y is not None]
    if not ys2:
        return
    ax.plot(xs2, ys2, color=color, linewidth=2, marker="o", markersize=3.5, label=label)


def plot(history, out_path):
    ep = [r["epoch"] for r in history]
    is_cakd = any(r.get("gl_loss") is not None for r in history)

    if is_cakd:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        # Panel 1: loss tong (train)
        _line(axes[0], ep, [r.get("train_loss") for r in history], BLUE, "train loss")
        _style(axes[0], "Loss tổng (train)", ylabel="loss")
        axes[0].legend(frameon=False, fontsize=8)
        # Panel 2: cac loss thanh phan (log scale vi khac thang do)
        _line(axes[1], ep, [r.get("cls_loss") for r in history], BLUE, "cls")
        _line(axes[1], ep, [r.get("pca_loss") for r in history], ORANGE, "pca")
        _line(axes[1], ep, [r.get("gl_loss") for r in history], GREEN, "gl")
        _line(axes[1], ep, [r.get("gan_loss") for r in history], PURPLE, "gan")
        axes[1].set_yscale("log")
        _style(axes[1], "Các loss thành phần (log)", ylabel="loss")
        axes[1].legend(frameon=False, fontsize=8)
        # Panel 3: accuracy
        _line(axes[2], ep, [r.get("train_acc1") for r in history], BLUE, "train acc@1")
        _line(axes[2], ep, [r.get("test_acc1") for r in history], ORANGE, "val acc@1")
        _style(axes[2], "Accuracy", ylabel="acc@1 (%)")
        axes[2].legend(frameon=False, fontsize=8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        _line(axes[0], ep, [r.get("train_loss") for r in history], BLUE, "train loss")
        _style(axes[0], "Loss (train)", ylabel="loss")
        axes[0].legend(frameon=False, fontsize=8)
        _line(axes[1], ep, [r.get("train_acc1") for r in history], BLUE, "train acc@1")
        _line(axes[1], ep, [r.get("test_acc1") for r in history], ORANGE, "val acc@1")
        _style(axes[1], "Accuracy", ylabel="acc@1 (%)")
        axes[1].legend(frameon=False, fontsize=8)

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
    args = p.parse_args()

    with open(args.history) as f:
        history = json.load(f)
    if not history:
        raise SystemExit("File history rỗng — chưa có epoch nào được ghi.")

    out = args.out or os.path.splitext(args.history)[0] + ".png"
    plot(history, out)


if __name__ == "__main__":
    main()
