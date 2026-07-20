"""Danh gia CHI TIET tu 1 checkpoint da train — KHONG can train lai.

Chay 1 luot inference tren tap val/test (ImageFolder), thu y_true vs y_pred roi tinh:
  - accuracy tong
  - precision / recall / f1 / support cho TUNG lop
  - macro avg + weighted avg
  - ma tran nham lan (counts + normalized)
Xuat: metrics.json (so lieu) + confusion_matrix.png (hinh).

Dung (chay trong thu muc CAKD/):
    python eval_metrics.py \
      --data-path /kaggle/working/data_test \
      --checkpoint /kaggle/working/results/checkpoint.pth \
      --student-arch resnet18 \
      --out-dir /kaggle/working/results

Ghi chu:
  - --data-path phai co thu muc con `val/` dang ImageFolder (class/anh...). Tap test cua ban da
    duoc copy sang `val/` o O 6 nen tro thang vao do.
  - Mac dinh dung trong so EMA (khop voi so 'best' luc train co --model-ema). Doi --weights model
    de dung trong so thuong.
"""
import argparse
import json
import os
import sys

# Cho phep chay tu bat ky dau: them thu muc chua file nay (CAKD/) vao path de import model/new_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torchvision
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import new_utils
from models.resnet_cakd import resnet18_cakd, resnet50_cakd
from models.mobilenet_cakd import mobilenetv3_small_cakd
from models.vit_cakd import build_teacher

STUDENT = {"resnet18": resnet18_cakd, "resnet50": resnet50_cakd,
           "mobilenetv3_small": mobilenetv3_small_cakd}


def _load_state(model, ckpt, prefer_ema):
    """Nap trong so vao model. prefer_ema=True -> uu tien 'model_ema' (bo tien to 'module.',
    bo 'n_averaged'); neu khong co thi rot ve 'model'."""
    if prefer_ema and "model_ema" in ckpt:
        raw = ckpt["model_ema"]
        sd = {k.replace("module.", "", 1): v for k, v in raw.items()
              if k != "n_averaged"}
        model.load_state_dict(sd)
        return "EMA"
    model.load_state_dict(ckpt["model"])
    return "model"


@torch.inference_mode()
def collect_preds(model, loader, device):
    y_true, y_pred = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)[0]           # student tra 4-tuple, logits o vi tri 0
        preds = logits.argmax(1).cpu()
        y_pred.append(preds)
        y_true.append(targets)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def plot_confusion(cm, classes, out_path):
    """Ve 2 ma tran canh nhau: counts (trai) + normalized theo hang/recall (phai)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(6.2 * 2, 5.4))
    for ax, mat, title, fmt, cmap in (
        (axes[0], cm, "Confusion matrix (counts)", "d", "Blues"),
        (axes[1], cm_norm, "Chuẩn hoá theo hàng (recall)", ".2f", "Greens"),
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max() if fmt == "d" else 1.0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Dự đoán", fontsize=10)
        ax.set_ylabel("Thực tế", fontsize=10)
        thr = mat.max() / 2 if fmt == "d" else 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=10,
                        color="white" if v > thr else "#222222")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Đã lưu ma trận nhầm lẫn: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Đánh giá chi tiết + ma trận nhầm lẫn từ checkpoint")
    p.add_argument("--data-path", required=True,
                   help="thư mục có val/ dạng ImageFolder (tập test đã copy sang val/)")
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint cần đánh giá (student: results/checkpoint.pth; "
                        "teacher: teacher_3cls.pth)")
    p.add_argument("--model", default="student", choices=["student", "teacher"],
                   help="đánh giá student (ResNet CAKD) hay teacher (ViT-B/16)")
    p.add_argument("--student-arch", default="mobilenetv3_small", choices=list(STUDENT),
                   help="kiến trúc student (bỏ qua nếu --model teacher)")
    p.add_argument("--weights", default="ema", choices=["ema", "model"],
                   help="student: EMA (mặc định, khớp best) hay model thường. Teacher luôn dùng model")
    p.add_argument("--crop-size", default=224, type=int)
    p.add_argument("--resize-size", default=224, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--out-dir", default=".", help="nơi lưu metrics.json + confusion_matrix.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Data: ImageFolder val/ + preprocessing giống lúc train
    val_dir = os.path.join(args.data_path, "val")
    if not os.path.isdir(val_dir):
        raise SystemExit(f"Không thấy {val_dir} — cần thư mục val/ dạng ImageFolder.")
    tf = new_utils.ClassificationPresetEval(crop_size=args.crop_size,
                                            resize_size=args.resize_size)
    ds = torchvision.datasets.ImageFolder(val_dir, tf)
    classes = ds.classes
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(f"Số ảnh: {len(ds)} | Lớp: {classes}")

    # 2) Model + nạp trọng số
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.model == "teacher":
        model = build_teacher(len(classes), pretrained=False).to(device)
        model.eval()
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        tag, used = "teacher", "model"
        print(f"Đã nạp teacher ViT-B/16 ({len(classes)} lớp) từ {args.checkpoint}")
    else:
        model = STUDENT[args.student_arch](num_classes=len(classes), pretrained=False).to(device)
        model.eval()
        used = _load_state(model, ckpt, prefer_ema=(args.weights == "ema"))
        tag = f"student_{args.student_arch}"
        print(f"Đã nạp student {args.student_arch}, trọng số: {used}")

    # 3) Inference thu y_true / y_pred
    y_true, y_pred = collect_preds(model, loader, device)

    # 4) Tính metric
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    # 5) In gọn ra màn hình
    print(f"\n=== Accuracy tổng: {acc*100:.2f}% ===")
    print(f"{'lớp':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c in classes:
        r = report[c]
        print(f"{c:<10}{r['precision']:>10.4f}{r['recall']:>10.4f}"
              f"{r['f1-score']:>10.4f}{int(r['support']):>10}")
    m = report["macro avg"]
    w = report["weighted avg"]
    print(f"{'macro':<10}{m['precision']:>10.4f}{m['recall']:>10.4f}{m['f1-score']:>10.4f}")
    print(f"{'weighted':<10}{w['precision']:>10.4f}{w['recall']:>10.4f}{w['f1-score']:>10.4f}")
    print("\nMa trận nhầm lẫn (hàng=thực tế, cột=dự đoán):")
    print("        " + "  ".join(f"{c[:6]:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:6]:>6}  " + "  ".join(f"{int(v):>6}" for v in cm[i]))

    # 6) Lưu JSON + PNG (tên kèm tag để student/teacher không ghi đè nhau)
    out_json = os.path.join(args.out_dir, f"metrics_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "arch": "vit_b_16" if args.model == "teacher" else args.student_arch,
            "weights": used,
            "num_images": len(ds),
            "classes": classes,
            "accuracy": acc,
            "per_class": {c: report[c] for c in classes},
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "confusion_matrix": cm.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu số liệu: {out_json}")
    plot_confusion(cm, classes, os.path.join(args.out_dir, f"confusion_matrix_{tag}.png"))


if __name__ == "__main__":
    main()
