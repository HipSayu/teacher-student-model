"""Bao cao truc quan qua trinh train tu cac file history JSON.

Khac plot_training.py (ve nhanh 1 run): script nay ve bao cao day du cho MOT hoac
NHIEU run cung luc, gom ca learning rate, va sinh them bang so lieu (CSV/Markdown)
de doi chieu voi bieu do.

Dung:
    python tools/plot_report.py teacher_train.json teacher_student.json --outdir img/report
    python tools/plot_report.py teacher_student.json --labels "CAKD student"

Moi history la list dict {epoch, train_loss, train_acc1, test_acc1, lr,
[cls_loss, pca_loss, gl_loss, gan_loss]}. Run co gl_loss => CAKD (teacher-student),
khong co => train teacher thuong.

San pham trong --outdir:
    <run>.png       bieu do tung run (loss / loss thanh phan / accuracy / lr)
    compare.png     so sanh cac run (chi khi co >= 2 run)
    <run>.csv       bang so lieu tung epoch (table view di kem bieu do)
    summary.md      bang tom tat: best acc, epoch tot nhat, acc cuoi, overfit gap
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # backend headless (Kaggle/CI khong co man hinh)
import matplotlib.pyplot as plt

# --- Bang mau (da validate: dai sang, chroma, tach mau cho nguoi mu mau, tuong phan) ---
BLUE = "#2a78d6"    # train / run 1
ORANGE = "#eb6834"  # validation
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"  # run 2
RED = "#e34948"

# Mau chu va khung (khong bao gio to chu bang mau cua duong)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Mau cap cho tung run khi so sanh (theo thu tu co dinh, khong xoay vong)
RUN_COLORS = [BLUE, VIOLET, ORANGE, AQUA]

# Ten hien thi cua cac loss thanh phan trong CAKD
COMPONENTS = [
    ("cls_loss", "cls", BLUE),
    ("pca_loss", "pca", AQUA),
    ("gl_loss", "gl", YELLOW),
    ("gan_loss", "gan", VIOLET),
]


# ---------------------------------------------------------------- helpers ----
def _clean(epochs, values):
    """Bo cac epoch co gia tri None (vd teacher khong co cls/pca/gl/gan)."""
    pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
    return [e for e, _ in pairs], [v for _, v in pairs]


def _style(ax, title, ylabel="", xlabel="epoch"):
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)  # luoi net lien, mo nhat
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def _line(ax, epochs, values, color, label, marker_limit=30):
    """Duong 2px; chi ve marker khi it epoch, nhieu epoch marker se thanh nhieu."""
    xs, ys = _clean(epochs, values)
    if not ys:
        return None
    marker = "o" if len(xs) <= marker_limit else None
    ax.plot(xs, ys, color=color, linewidth=2, label=label,
            marker=marker, markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
    return xs[-1], ys[-1]


def _end_label(ax, point, color, text):
    """Direct-label o cuoi duong: bat buoc voi cac mau tuong phan thap (aqua/yellow)."""
    if point is None:
        return
    x, y = point
    ax.annotate(f" {text}", xy=(x, y), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK_2, va="center", ha="left",
                annotation_clip=False)


def _mark_best(ax, epochs, values, color):
    """Danh dau epoch co val acc cao nhat (diem noi bat, khong ghi so len moi diem)."""
    xs, ys = _clean(epochs, values)
    if not ys:
        return
    best_i = max(range(len(ys)), key=lambda i: ys[i])
    bx, by = xs[best_i], ys[best_i]
    ax.plot([bx], [by], marker="o", markersize=9, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5, linestyle="none")
    # Chua nhan o tren diem => can khoang trong phia tren, neu khong se de len
    # duong train hoac tieu de panel.
    ax.margins(y=0.18)
    ax.annotate(f"best {by:.2f}% @ep{bx}", xy=(bx, by), xytext=(0, 12),
                textcoords="offset points", fontsize=8, color=INK,
                fontweight="bold", ha="center")


def _lr_scale(ax, lrs):
    """LR luon o panel rieng — khong bao gio dung truc y thu hai chung voi loss."""
    vals = [v for v in lrs if v is not None and v > 0]
    if vals and max(vals) / min(vals) > 20:
        ax.set_yscale("log")


def is_cakd(history):
    return any(r.get("gl_loss") is not None for r in history)


# ------------------------------------------------------------- figure: run ----
def plot_run(history, label, out_path):
    ep = [r["epoch"] for r in history]
    cakd = is_cakd(history)

    ncols = 4 if cakd else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.3 * ncols, 4.4), facecolor=SURFACE)
    fig.suptitle(f"{label} — {len(history)} epoch", fontsize=13, fontweight="bold",
                 color=INK, y=1.02)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    i = 0
    # 1. Loss tong
    _line(axes[i], ep, [r.get("train_loss") for r in history], BLUE, "train loss")
    _style(axes[i], "Loss tổng (train)", ylabel="loss")
    axes[i].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    i += 1

    # 2. Cac loss thanh phan (chi CAKD) — log scale vi khac thang do
    if cakd:
        for key, name, color in COMPONENTS:
            pt = _line(axes[i], ep, [r.get(key) for r in history], color, name)
            _end_label(axes[i], pt, color, name)  # direct label: bu tuong phan thap
        axes[i].set_yscale("log")
        _style(axes[i], "Các loss thành phần (log)", ylabel="loss")
        axes[i].legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=2)
        i += 1

    # 3. Accuracy
    _line(axes[i], ep, [r.get("train_acc1") for r in history], BLUE, "train acc@1")
    test = [r.get("test_acc1") for r in history]
    _line(axes[i], ep, test, ORANGE, "val acc@1")
    _mark_best(axes[i], ep, test, ORANGE)
    _style(axes[i], "Accuracy", ylabel="acc@1 (%)")
    axes[i].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    i += 1

    # 4. Learning rate — panel rieng
    lrs = [r.get("lr") for r in history]
    _line(axes[i], ep, lrs, AQUA, "lr")
    _lr_scale(axes[i], lrs)
    _style(axes[i], "Learning rate", ylabel="lr")
    axes[i].legend(frameon=False, fontsize=8, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  → {out_path}")


# --------------------------------------------------------- figure: compare ----
def plot_compare(runs, out_path):
    """So sanh nhieu run.

    Chi so sanh accuracy: train_loss cua teacher (chi cross-entropy) va cua student
    (CE + pca + gl + gan) la HAI DAI LUONG KHAC NHAU — dat chung mot truc se tao ra
    mot so sanh khong co that. Accuracy cung don vi (%) nen so duoc.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    fig.suptitle("So sánh các run", fontsize=13, fontweight="bold", color=INK, y=1.02)

    for idx, (label, history) in enumerate(runs):
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        ep = [r["epoch"] for r in history]

        test = [r.get("test_acc1") for r in history]
        # marker_limit=0: khong marker, de cac run co style dong nhat
        _line(axes[0], ep, test, color, label, marker_limit=0)
        _mark_best(axes[0], ep, test, color)
        _line(axes[1], ep, [r.get("train_acc1") for r in history], color, label,
              marker_limit=0)

    _style(axes[0], "Validation acc@1", ylabel="acc@1 (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    _style(axes[1], "Train acc@1", ylabel="acc@1 (%)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  → {out_path}")


# ------------------------------------------------------------------ tables ----
FIELDS = ["epoch", "train_loss", "train_acc1", "test_acc1",
          "cls_loss", "pca_loss", "gl_loss", "gan_loss", "lr"]


def write_csv(history, out_path):
    lines = [",".join(FIELDS)]
    for r in history:
        lines.append(",".join(
            "" if r.get(k) is None else str(r.get(k)) for k in FIELDS))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {out_path}")


def stats(history):
    test = [(r["epoch"], r["test_acc1"]) for r in history
            if r.get("test_acc1") is not None]
    best_ep, best_acc = max(test, key=lambda t: t[1]) if test else (None, None)
    last = history[-1]
    gap = None
    if last.get("train_acc1") is not None and last.get("test_acc1") is not None:
        gap = last["train_acc1"] - last["test_acc1"]
    return {
        "epochs": len(history),
        "best_acc": best_acc,
        "best_epoch": best_ep,
        "last_acc": last.get("test_acc1"),
        "last_train_acc": last.get("train_acc1"),
        "last_loss": last.get("train_loss"),
        "gap": gap,
    }


def write_summary(runs, out_path):
    rows = ["| Run | Loại | Epoch | Best val acc@1 | Tại epoch | Val acc cuối | "
            "Train acc cuối | Gap (train−val) |",
            "|---|---|---|---|---|---|---|---|"]
    for label, history in runs:
        s = stats(history)
        kind = "CAKD (teacher→student)" if is_cakd(history) else "Teacher"
        rows.append(
            f"| {label} | {kind} | {s['epochs']} | **{s['best_acc']:.2f}%** | "
            f"{s['best_epoch']} | {s['last_acc']:.2f}% | {s['last_train_acc']:.2f}% | "
            f"{s['gap']:+.2f} |")
    table = "\n".join(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Tóm tắt quá trình train\n\n" + table + "\n")
    print(f"  → {out_path}")
    print()
    print(table)


# -------------------------------------------------------------------- main ----
def main():
    # Console Windows mac dinh cp1252, khong in duoc tieng Viet
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(
        description="Vẽ báo cáo quá trình train từ một hoặc nhiều history JSON")
    p.add_argument("history", nargs="+", help="các file history JSON")
    p.add_argument("--labels", nargs="*", default=None,
                   help="tên hiển thị cho từng file (mặc định lấy theo tên file)")
    p.add_argument("--outdir", default="img/report", help="thư mục xuất kết quả")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.history):
        raise SystemExit("Số --labels phải bằng số file history.")

    runs = []
    for i, path in enumerate(args.history):
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
        if not history:
            raise SystemExit(f"{path}: history rỗng — chưa có epoch nào được ghi.")
        label = args.labels[i] if args.labels else \
            os.path.splitext(os.path.basename(path))[0]
        runs.append((label, history))

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Xuất báo cáo vào: {args.outdir}")

    for label, history in runs:
        slug = label.replace(" ", "_").replace("/", "-")
        plot_run(history, label, os.path.join(args.outdir, f"{slug}.png"))
        write_csv(history, os.path.join(args.outdir, f"{slug}.csv"))

    if len(runs) >= 2:
        plot_compare(runs, os.path.join(args.outdir, "compare.png"))

    write_summary(runs, os.path.join(args.outdir, "summary.md"))


if __name__ == "__main__":
    main()
