"""So sanh DU DOAN TUNG ANH giua baseline / CAKD student / teacher tren cung tap test.

Muc dich: tim cac vi du dinh tinh cho bao cao —
  - anh baseline SAI nhung CAKD DUNG  -> "duoc CAKD sua"       (fixed_by_cakd.png)
  - anh baseline DUNG nhung CAKD SAI  -> "bi CAKD lam hong"    (broken_by_cakd.png)
Kem bang doi chieu 2x2 + kiem dinh McNemar de biet chenh lech co y nghia thong ke khong.

Chay trong thu muc CAKD/:
    python compare_predictions.py \
      --data-path   /kaggle/working/data_test \
      --baseline-ckpt /kaggle/working/baseline_resnet18/resnet18_baseline_best.pth \
      --student-ckpt  /kaggle/working/results/checkpoint.pth \
      --teacher-ckpt  /kaggle/working/teacher_3cls_best.pth \
      --out-dir /kaggle/working/qualitative

Ghi chu:
  - --data-path phai co thu muc con `val/` dang ImageFolder (giong eval_metrics.py).
  - Ca 3 model dung CHUNG 1 DataLoader (shuffle=False) nen thu tu anh khop tuyet doi.
  - Xep hang vi du: uu tien truong hop TEACHER CUNG DUNG (bang chung tri thuc truyen tu
    teacher sang student), sau do theo do tang xac suat lop dung (CAKD - baseline).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import new_utils
from eval_metrics import STUDENT, build_plain, _load_state
from models.vit_cakd import build_teacher

OK_COLOR, BAD_COLOR = "#1a7f37", "#c0392b"


@torch.inference_mode()
def collect_probs(model, loader, device, tuple_output):
    """Chay 1 luot inference, tra ve mang xac suat softmax (N, num_classes)."""
    out_all = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        logits = out[0] if tuple_output else out
        out_all.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(out_all).numpy()


def load_model(kind, arch, ckpt_path, num_classes, device, prefer_ema=True):
    """kind: 'baseline' (torchvision thuong) | 'student' (CAKD) | 'teacher' (ViT).
    Tra ve (model da eval, tuple_output, mo ta trong so da dung)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if kind == "teacher":
        model = build_teacher(num_classes, pretrained=False).to(device).eval()
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        return model, True, "model"
    if kind == "baseline":
        model = build_plain(arch, num_classes).to(device).eval()
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        return model, False, "model"   # model thuong tra thang logits
    model = STUDENT[arch](num_classes=num_classes, pretrained=False).to(device).eval()
    used = _load_state(model, ckpt, prefer_ema=prefer_ema)
    return model, True, used


def load_for_display(path, crop_size, resize_size):
    """Nap lai anh goc, ap DUNG resize+crop nhu luc inference (bo normalize) de ve."""
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, resize_size, interpolation=InterpolationMode.BILINEAR)
    return TF.center_crop(img, [crop_size, crop_size])


def draw_grid(records, classes, title, out_path, crop_size, resize_size, has_teacher):
    """Ve luoi anh kem nhan that + du doan tung model (xanh=dung, do=sai)."""
    n = len(records)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    row_h = 4.6 if has_teacher else 4.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, row_h * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for k, r in enumerate(records):
        ax = axes[k // ncols][k % ncols]
        ax.imshow(load_for_display(r["path"], crop_size, resize_size))
        ax.axis("off")
        ax.set_title(f"Thực tế: {classes[r['true']]}", fontsize=11, fontweight="bold", pad=6)

        lines = [("Baseline", r["baseline_pred"], r["baseline_conf"]),
                 ("CAKD", r["student_pred"], r["student_conf"])]
        if has_teacher:
            lines.append(("Teacher", r["teacher_pred"], r["teacher_conf"]))
        for j, (name, pred, conf) in enumerate(lines):
            correct = pred == r["true"]
            mark = "✓" if correct else "✗"
            ax.text(0.5, -0.055 - 0.075 * j,
                    f"{name}: {classes[pred]} ({conf:.2f}) {mark}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=10,
                    color=OK_COLOR if correct else BAD_COLOR,
                    fontweight="bold" if name == "CAKD" else "normal")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {out_path}")


def mcnemar_pvalue(n_fixed, n_broken):
    """Kiem dinh McNemar (exact, 2 phia) tren cac cap bat dong. None neu thieu scipy."""
    try:
        from scipy.stats import binomtest
    except ImportError:
        return None
    if n_fixed + n_broken == 0:
        return 1.0
    return binomtest(n_fixed, n_fixed + n_broken, 0.5).pvalue


def main():
    p = argparse.ArgumentParser(description="So sánh dự đoán từng ảnh: baseline vs CAKD vs teacher")
    p.add_argument("--data-path", required=True, help="thư mục có val/ dạng ImageFolder")
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--baseline-arch", default="resnet18", choices=list(STUDENT))
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--student-arch", default="resnet18", choices=list(STUDENT))
    p.add_argument("--student-weights", default="ema", choices=["ema", "model"])
    p.add_argument("--teacher-ckpt", default=None, help="tùy chọn — thêm dòng teacher vào hình")
    p.add_argument("--num-examples", default=8, type=int, help="số ảnh tối đa mỗi lưới")
    p.add_argument("--crop-size", default=224, type=int)
    p.add_argument("--resize-size", default=224, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--out-dir", default="qualitative")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    val_dir = os.path.join(args.data_path, "val")
    if not os.path.isdir(val_dir):
        raise SystemExit(f"Không thấy {val_dir} — cần thư mục val/ dạng ImageFolder.")
    tf = new_utils.ClassificationPresetEval(crop_size=args.crop_size, resize_size=args.resize_size)
    ds = torchvision.datasets.ImageFolder(val_dir, tf)
    classes = ds.classes
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    y_true = np.array([lbl for _, lbl in ds.samples])
    paths = [pth for pth, _ in ds.samples]
    print(f"Số ảnh test: {len(ds)} | Lớp: {classes}")

    # --- Chạy 3 model trên CÙNG loader (shuffle=False -> thứ tự ảnh khớp tuyệt đối) ---
    m, tup, _ = load_model("baseline", args.baseline_arch, args.baseline_ckpt, len(classes), device)
    print(f"Đã nạp baseline {args.baseline_arch}")
    p_base = collect_probs(m, loader, device, tup)
    del m
    torch.cuda.empty_cache()

    m, tup, used = load_model("student", args.student_arch, args.student_ckpt, len(classes),
                              device, prefer_ema=(args.student_weights == "ema"))
    print(f"Đã nạp CAKD student {args.student_arch}, trọng số: {used}")
    p_stu = collect_probs(m, loader, device, tup)
    del m
    torch.cuda.empty_cache()

    p_tea = None
    if args.teacher_ckpt:
        m, tup, _ = load_model("teacher", None, args.teacher_ckpt, len(classes), device)
        print("Đã nạp teacher ViT-B/16")
        p_tea = collect_probs(m, loader, device, tup)
        del m
        torch.cuda.empty_cache()

    pred_base, pred_stu = p_base.argmax(1), p_stu.argmax(1)
    ok_base, ok_stu = pred_base == y_true, pred_stu == y_true

    # --- Bảng đối chiếu 2x2 ---
    fixed_mask = (~ok_base) & ok_stu      # CAKD sửa được
    broken_mask = ok_base & (~ok_stu)     # CAKD làm hỏng
    n_fixed, n_broken = int(fixed_mask.sum()), int(broken_mask.sum())
    n_both_ok, n_both_bad = int((ok_base & ok_stu).sum()), int(((~ok_base) & (~ok_stu)).sum())
    pval = mcnemar_pvalue(n_fixed, n_broken)

    print(f"\n=== Đối chiếu trên {len(ds)} ảnh test ===")
    print(f"  baseline ĐÚNG & CAKD ĐÚNG : {n_both_ok}")
    print(f"  baseline SAI  & CAKD ĐÚNG : {n_fixed}   <- CAKD SỬA ĐƯỢC")
    print(f"  baseline ĐÚNG & CAKD SAI  : {n_broken}   <- CAKD LÀM HỎNG")
    print(f"  baseline SAI  & CAKD SAI  : {n_both_bad}")
    print(f"  accuracy: baseline {ok_base.mean()*100:.2f}%  |  CAKD {ok_stu.mean()*100:.2f}%"
          f"  |  chênh {(ok_stu.mean()-ok_base.mean())*100:+.2f} điểm")
    if pval is not None:
        print(f"  McNemar (exact, 2 phía) trên {n_fixed + n_broken} cặp bất đồng: p = {pval:.4g}")

    # --- Gom bản ghi từng ảnh ---
    def make_records(mask):
        out = []
        for i in np.flatnonzero(mask):
            r = {"path": paths[i], "true": int(y_true[i]),
                 "baseline_pred": int(pred_base[i]), "baseline_conf": float(p_base[i, pred_base[i]]),
                 "student_pred": int(pred_stu[i]), "student_conf": float(p_stu[i, pred_stu[i]]),
                 # độ tăng xác suất gán cho LỚP ĐÚNG — thước đo "cải thiện thật" của ví dụ này
                 "gain_true_class": float(p_stu[i, y_true[i]] - p_base[i, y_true[i]])}
            if p_tea is not None:
                r["teacher_pred"] = int(p_tea[i].argmax())
                r["teacher_conf"] = float(p_tea[i].max())
                r["teacher_correct"] = bool(r["teacher_pred"] == r["true"])
            out.append(r)
        # ưu tiên ví dụ teacher CŨNG đúng (bằng chứng truyền tri thức), rồi tới mức cải thiện lớn nhất
        out.sort(key=lambda r: (r.get("teacher_correct", False), r["gain_true_class"]), reverse=True)
        return out

    fixed, broken = make_records(fixed_mask), make_records(broken_mask)

    out_json = os.path.join(args.out_dir, "compare_predictions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes, "num_images": len(ds),
            "accuracy": {"baseline": float(ok_base.mean()), "cakd": float(ok_stu.mean())},
            "agreement": {"both_correct": n_both_ok, "fixed_by_cakd": n_fixed,
                          "broken_by_cakd": n_broken, "both_wrong": n_both_bad},
            "mcnemar_pvalue": pval,
            "fixed_by_cakd": fixed, "broken_by_cakd": broken,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu danh sách đầy đủ: {out_json}")

    has_tea = p_tea is not None
    for recs, fname, title in (
        (fixed, "fixed_by_cakd.png", f"Baseline SAI → CAKD ĐÚNG ({n_fixed} ảnh trong tập test)"),
        (broken, "broken_by_cakd.png", f"Baseline ĐÚNG → CAKD SAI ({n_broken} ảnh trong tập test)"),
    ):
        if not recs:
            print(f"Không có ảnh nào thuộc nhóm '{fname}' — bỏ qua hình.")
            continue
        draw_grid(recs[:args.num_examples], classes, title,
                  os.path.join(args.out_dir, fname), args.crop_size, args.resize_size, has_tea)


if __name__ == "__main__":
    main()
