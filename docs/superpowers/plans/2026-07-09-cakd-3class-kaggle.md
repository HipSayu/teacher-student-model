# CAKD 3-Class (glass/paper/plastic) trên Kaggle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Làm cho project CAKD chạy được để distill teacher ViT-B/16 → student ResNet-50 phân loại 3 lớp glass/paper/plastic, với Data 2 (dạng `class/split/images`) trên Kaggle 1 GPU.

**Architecture:** 4 bước tuần tự giao tiếp qua file trên đĩa: (B1) reorg data → ImageFolder; (B0/B2 setup) chép 3 file "độ" + verify; (B2) fine-tune teacher xuống 3 lớp → `teacher_3cls.pth`; (B3) sửa `dist_train_cakd.py` để nạp teacher 3 lớp + student pretrained → distill → `checkpoint.pth`; (B4) đánh giá test. Không viết lại kiến trúc CAKD, chỉ thêm/sửa tối thiểu.

**Tech Stack:** Python 3, PyTorch 1.12.0 + torchvision 0.13.0 (cu113), einops, torchrun (1 process), Kaggle Notebook GPU.

## Global Constraints

- **Số lớp = 3**, tên lớp cố định thứ tự alphabet do `ImageFolder`: `glass=0, paper=1, plastic=2`.
- **PyTorch bị ghim `torch==1.12.0+cu113`, `torchvision==0.13.0`** — 3 file "độ" (`resnet.py`, `vision_transformer.py`, `functional.py`) chỉ khớp API nội bộ của bản này.
- **Bắt buộc GPU**: `GLProj.forward` hardcode `.to('cuda')` (`CAKD/cakd_modified_files/resnet.py:207`).
- **Teacher phải là 3 lớp trước khi distill** — nếu không `gl_loss`/logits crash (lệch shape 3 vs 1000).
- **Student nạp pretrained `strict=False` và bị ép `num_classes=1000`** → phải thay `model.fc = nn.Linear(2048, 3)` sau khi tạo.
- **Chạy 1 GPU** qua `torchrun --nproc_per_node=1` để `init_distributed_mode` khởi tạo process group (evaluate an toàn).
- **Kaggle input read-only** → mọi output ghi vào `/kaggle/working`.
- Không đụng công thức 4 loss / vòng GAN / EMA của CAKD.

---

## File Structure

| File | Trách nhiệm | Tạo/Sửa |
|---|---|---|
| `tools/reorg_to_imagefolder.py` | Reorg Data 2 (`class/split/images`) → ImageFolder (`split/class`), bỏ `labels/`, in thống kê | Tạo |
| `tools/tests/test_reorg_to_imagefolder.py` | Unit test cho reorg (chạy local, không cần GPU) | Tạo |
| `CAKD/setup_kaggle.py` | Định vị + sao lưu + chép 3 file "độ" vào site-packages; hàm `verify()` forward thử | Tạo |
| `CAKD/tests/test_setup_kaggle.py` | Unit test cho phần copy (temp dir, không cần torch 1.12) | Tạo |
| `CAKD/dist_train_teacher.py` | GĐ1: fine-tune ViT-B/16 → 3 lớp, lưu `teacher_3cls.pth` | Tạo |
| `CAKD/dist_train_cakd.py` | GĐ2: nạp teacher 3 lớp + student pretrained + cờ distill-start/ramp | Sửa |
| `experiments/run_teacher_kaggle.sh` | Lệnh chạy GĐ1 (1 GPU) | Tạo |
| `experiments/run_cakd_kaggle.sh` | Lệnh chạy GĐ2 (1 GPU) | Tạo |
| `KAGGLE_HUONGDAN.md` | Thứ tự các ô notebook Kaggle (B0→B4) + demo suy luận | Tạo |

---

## Task 1: Công cụ reorg Data 2 → ImageFolder

**Files:**
- Create: `tools/reorg_to_imagefolder.py`
- Test: `tools/tests/test_reorg_to_imagefolder.py`

**Interfaces:**
- Produces: `reorg_to_imagefolder(src: str, dst: str, classes: list[str], splits: list[str], copy: bool = False) -> dict[str, dict[str, int]]` — trả về thống kê `{split: {class: so_anh}}`. Ảnh lấy từ `src/<class>/<split>/images/`, ghi ra `dst/<split>/<class>/`. `labels/` bị bỏ qua. Split thiếu → cảnh báo, không lỗi.

- [ ] **Step 1: Viết test thất bại**

```python
# tools/tests/test_reorg_to_imagefolder.py
import os
import pytest
from tools.reorg_to_imagefolder import reorg_to_imagefolder

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff")  # vài byte giả làm "ảnh"


def _make_src(root):
    # class/split/images/*.jpg  + labels/*.txt (phải bị bỏ qua)
    layout = {
        "glass":   {"train": 3, "val": 1, "test": 1},
        "paper":   {"train": 2, "val": 1, "test": 1},
        "plastic": {"train": 2, "val": 1},  # thiếu test cố ý
    }
    for cls, splits in layout.items():
        for split, n in splits.items():
            for i in range(n):
                _touch(os.path.join(root, cls, split, "images", f"{cls}_{split}_{i}.jpg"))
                _touch(os.path.join(root, cls, split, "labels", f"{cls}_{split}_{i}.txt"))
    return layout


def test_reorg_creates_imagefolder_layout(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_src(str(src))

    stats = reorg_to_imagefolder(
        str(src), str(dst),
        classes=["glass", "paper", "plastic"],
        splits=["train", "val", "test"],
        copy=True,  # copy để test không phụ thuộc quyền symlink trên Windows
    )

    # đúng số ảnh, bỏ qua labels
    assert stats["train"] == {"glass": 3, "paper": 2, "plastic": 2}
    assert stats["val"] == {"glass": 1, "paper": 1, "plastic": 1}
    # plastic thiếu test -> 0
    assert stats["test"]["glass"] == 1
    assert stats["test"]["plastic"] == 0

    # cấu trúc ImageFolder: dst/split/class/*.jpg, KHÔNG có tầng images/ hay labels/
    train_glass = dst / "train" / "glass"
    assert train_glass.is_dir()
    files = list(train_glass.iterdir())
    assert len(files) == 3
    assert all(f.suffix in IMG_EXTS for f in files)
    assert not (dst / "train" / "glass" / "images").exists()
    assert not (dst / "train" / "glass" / "labels").exists()
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `python -m pytest tools/tests/test_reorg_to_imagefolder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.reorg_to_imagefolder'`

- [ ] **Step 3: Viết implementation tối thiểu**

```python
# tools/reorg_to_imagefolder.py
"""Reorg Data 2 (class/split/images/*) -> ImageFolder (split/class/*), bo qua labels/.

Vi sao: dist_train_*.py dung torchvision.datasets.ImageFolder, doi hoi layout
    <data_path>/<split>/<class>/*.jpg
trong khi dataset goc la
    <src>/<class>/<split>/images/*.jpg  (+ labels/*.txt cho detection)

Chay tren Kaggle:
    python tools/reorg_to_imagefolder.py --src /kaggle/input/<ten-dataset> \
        --dst /kaggle/working/data_if
Sau do dung --data-path /kaggle/working/data_if cho cac script train.
"""
import argparse
import os
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]


def reorg_to_imagefolder(src, dst, classes, splits, copy=False):
    """Tao dst/<split>/<class>/ tu src/<class>/<split>/images/. Tra ve thong ke."""
    stats = {}
    for split in splits:
        stats[split] = {}
        for cls in classes:
            img_dir = os.path.join(src, cls, split, "images")
            out_dir = os.path.join(dst, split, cls)
            images = _list_images(img_dir)
            if not images:
                # split hoac class thieu -> ghi nhan 0, canh bao neu ca thu muc split vang
                if not os.path.isdir(os.path.join(src, cls, split)):
                    print(f"[WARN] khong thay {src}/{cls}/{split} -> bo qua")
                stats[split][cls] = 0
                continue
            os.makedirs(out_dir, exist_ok=True)
            for name in images:
                s = os.path.join(img_dir, name)
                d = os.path.join(out_dir, name)
                if os.path.lexists(d):
                    os.remove(d)
                if copy:
                    shutil.copy2(s, d)
                else:
                    os.symlink(os.path.abspath(s), d)
            stats[split][cls] = len(images)
    return stats


def _print_stats(stats):
    print("\n===== THONG KE ANH =====")
    for split, per_cls in stats.items():
        total = sum(per_cls.values())
        detail = ", ".join(f"{c}={n}" for c, n in per_cls.items())
        print(f"  {split:5s}: tong={total:5d}  ({detail})")
    print("========================\n")


def main():
    p = argparse.ArgumentParser(description="Reorg class/split/images -> ImageFolder")
    p.add_argument("--src", required=True, help="thu muc goc chua <class>/<split>/images")
    p.add_argument("--dst", default="/kaggle/working/data_if", help="thu muc ImageFolder dau ra")
    p.add_argument("--classes", nargs="+", default=["glass", "paper", "plastic"])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--copy", action="store_true", help="copy that thay vi symlink")
    args = p.parse_args()

    stats = reorg_to_imagefolder(args.src, args.dst, args.classes, args.splits, args.copy)
    _print_stats(stats)
    print(f"Xong. Dung --data-path {args.dst} cho cac script train.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Tạo `tools/tests/__init__.py` và `tools/__init__.py` rỗng để import chạy**

```bash
touch tools/__init__.py tools/tests/__init__.py
```

- [ ] **Step 5: Chạy test để xác nhận PASS**

Run: `python -m pytest tools/tests/test_reorg_to_imagefolder.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add tools/reorg_to_imagefolder.py tools/tests/ tools/__init__.py
git commit -m "feat: reorg Data 2 (class/split/images) sang ImageFolder"
```

---

## Task 2: Setup môi trường Kaggle (chép 3 file độ + verify)

**Files:**
- Create: `CAKD/setup_kaggle.py`
- Test: `CAKD/tests/test_setup_kaggle.py`

**Interfaces:**
- Consumes: 3 file trong `CAKD/cakd_modified_files/` (`resnet.py`, `vision_transformer.py`, `functional.py`).
- Produces:
  - `install_modified_files(cakd_dir: str, tv_models_dir: str, torch_nn_dir: str, backup: bool = True) -> list[str]` — copy 3 file vào đích, sao lưu `.bak` nếu chưa có. Trả về danh sách đường dẫn đã ghi.
  - `verify() -> None` — build `resnet50_cakd(num_classes=3)` và `vit_b_16(num_classes=3)` trên cuda, forward 1 batch giả, assert 4-tuple & shape. Raise nếu sai. (Chỉ chạy được trên Kaggle GPU + torch 1.12.)

- [ ] **Step 1: Viết test thất bại (chỉ test phần copy, không cần torch 1.12)**

```python
# CAKD/tests/test_setup_kaggle.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from setup_kaggle import install_modified_files


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_install_copies_three_files_and_backs_up(tmp_path):
    cakd = tmp_path / "cakd_modified_files"
    _write(str(cakd / "resnet.py"), "NEW_RESNET")
    _write(str(cakd / "vision_transformer.py"), "NEW_VIT")
    _write(str(cakd / "functional.py"), "NEW_FUNC")

    tv_models = tmp_path / "site" / "torchvision" / "models"
    torch_nn = tmp_path / "site" / "torch" / "nn"
    _write(str(tv_models / "resnet.py"), "OLD_RESNET")
    _write(str(tv_models / "vision_transformer.py"), "OLD_VIT")
    _write(str(torch_nn / "functional.py"), "OLD_FUNC")

    written = install_modified_files(str(cakd), str(tv_models), str(torch_nn), backup=True)

    # 3 file dich da bi ghi de bang noi dung moi
    assert (tv_models / "resnet.py").read_text() == "NEW_RESNET"
    assert (tv_models / "vision_transformer.py").read_text() == "NEW_VIT"
    assert (torch_nn / "functional.py").read_text() == "NEW_FUNC"
    # da sao luu ban cu
    assert (tv_models / "resnet.py.bak").read_text() == "OLD_RESNET"
    assert (torch_nn / "functional.py.bak").read_text() == "OLD_FUNC"
    assert len(written) == 3
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `python -m pytest CAKD/tests/test_setup_kaggle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'setup_kaggle'`

- [ ] **Step 3: Viết implementation**

```python
# CAKD/setup_kaggle.py
"""Setup moi truong Kaggle cho CAKD: chep 3 file 'do' vao site-packages + verify.

Dung trong notebook Kaggle:
    !pip install torch==1.12.0+cu113 torchvision==0.13.0 \
        --extra-index-url https://download.pytorch.org/whl/cu113
    !pip install einops
    import setup_kaggle; setup_kaggle.main()   # copy + verify

Luu y: 3 file do chi khop torch 1.12 / torchvision 0.13. Chay tren ban khac se hong.
"""
import argparse
import os
import shutil


MODIFIED = {
    # ten file nguon trong cakd_modified_files -> (dich_key)
    "resnet.py": "tv_models",
    "vision_transformer.py": "tv_models",
    "functional.py": "torch_nn",
}


def install_modified_files(cakd_dir, tv_models_dir, torch_nn_dir, backup=True):
    """Copy 3 file do vao dich. Sao luu .bak neu chua co. Tra ve list duong dan da ghi."""
    dest_root = {"tv_models": tv_models_dir, "torch_nn": torch_nn_dir}
    written = []
    for fname, key in MODIFIED.items():
        src = os.path.join(cakd_dir, fname)
        dst = os.path.join(dest_root[key], fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Khong thay file nguon: {src}")
        if backup and os.path.isfile(dst) and not os.path.isfile(dst + ".bak"):
            shutil.copy2(dst, dst + ".bak")
        shutil.copy2(src, dst)
        written.append(dst)
        print(f"[OK] chep {fname} -> {dst}")
    return written


def locate_paths():
    """Tra ve (torchvision_models_dir, torch_nn_dir) cua ban torch dang cai."""
    import torch
    import torchvision
    tv_models = os.path.join(os.path.dirname(torchvision.models.__file__))
    torch_nn = os.path.dirname(torch.nn.__file__)
    return tv_models, torch_nn


def verify():
    """Build student + teacher 3 lop, forward thu 1 batch, assert 4-tuple & shape."""
    import torch
    import torchvision

    assert torch.cuda.is_available(), "Can GPU (GLProj hardcode .to('cuda'))"
    dev = "cuda"
    x = torch.randn(2, 3, 224, 224, device=dev)

    student = torchvision.models.resnet50_cakd(num_classes=3).to(dev)
    out = student(x)
    assert isinstance(out, tuple) and len(out) == 4, f"student tra ve {type(out)}"
    logits, attn, feat, token = out
    assert logits.shape == (2, 3), logits.shape
    assert attn[0].shape == (2, 196, 196), attn[0].shape

    teacher = torchvision.models.vit_b_16(num_classes=3).to(dev).eval()
    tout = teacher(x)
    assert isinstance(tout, tuple) and len(tout) == 4, f"teacher tra ve {type(tout)}"
    tlogits = tout[0]
    assert tlogits.shape == (2, 3), tlogits.shape
    print("[VERIFY] OK — student & teacher tra ve 4-tuple dung shape (3 lop).")


def main(cakd_dir=None):
    if cakd_dir is None:
        cakd_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cakd_modified_files")
    tv_models, torch_nn = locate_paths()
    install_modified_files(cakd_dir, tv_models, torch_nn, backup=True)
    verify()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cakd-dir", default=None, help="thu muc cakd_modified_files")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()
    cakd_dir = args.cakd_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cakd_modified_files")
    tv_models, torch_nn = locate_paths()
    install_modified_files(cakd_dir, tv_models, torch_nn, backup=True)
    if not args.no_verify:
        verify()
```

- [ ] **Step 4: Tạo `CAKD/tests/__init__.py` rỗng**

```bash
touch CAKD/tests/__init__.py
```

- [ ] **Step 5: Chạy test để xác nhận PASS**

Run: `python -m pytest CAKD/tests/test_setup_kaggle.py -v`
Expected: PASS (1 passed) — chỉ test phần copy, không import torch.

- [ ] **Step 6: Commit**

```bash
git add CAKD/setup_kaggle.py CAKD/tests/
git commit -m "feat: setup_kaggle chep 3 file do + verify forward 3 lop"
```

---

## Task 3: Script GĐ1 — fine-tune teacher ViT xuống 3 lớp

**Files:**
- Create: `CAKD/dist_train_teacher.py`
- Create: `experiments/run_teacher_kaggle.sh`

**Interfaces:**
- Consumes: `load_data` từ `dist_train_cakd.py` (đã có, đọc ImageFolder + augment); `new_utils` (MetricLogger, accuracy, init_distributed_mode...).
- Produces: file checkpoint `teacher_3cls.pth` = `torch.save({"model": teacher.state_dict(), ...})`. Teacher là `vit_b_16(num_classes=3)` với head đã fine-tune. Task 4 nạp bằng `teacher.load_state_dict(ckpt["model"])`.

**Ghi chú:** đây là script train ML — verify bằng **smoke run** trên Kaggle (2 epoch, ít ảnh) rồi kiểm tra log, không phải pytest.

- [ ] **Step 1: Viết `CAKD/dist_train_teacher.py`**

```python
# CAKD/dist_train_teacher.py
# =============================================================================
# GD1 — Fine-tune teacher ViT-B/16 (pretrain ImageNet) xuong 3 lop glass/paper/plastic.
# Vi sao: teacher goc xuat 1000 logits -> khong the distill logits cho student 3 lop
# (gl_loss se vo shape). Fine-tune teacher xuong 3 lop truoc, luu teacher_3cls.pth,
# roi dist_train_cakd.py nap teacher nay de distill.
#
# Chi dung CrossEntropy tren out[0] (forward ViT do tra 4-tuple, phan tu 0 = logits).
# Chay 1 GPU:  torchrun --nproc_per_node=1 dist_train_teacher.py --data-path <dir> ...
# =============================================================================
import datetime
import os
import time

import new_utils
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision.models import ViT_B_16_Weights

from dist_train_cakd import load_data  # tai su dung loader ImageFolder (DRY)


def build_teacher(num_classes, pretrained=True):
    """Tao ViT-B/16, nap pretrain ImageNet, thay head thanh num_classes lop."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.vit_b_16(weights=weights)  # head 1000 lop
    model.heads.head = nn.Linear(model.hidden_dim, num_classes)  # thay head -> 3 lop
    nn.init.zeros_(model.heads.head.bias)
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, scaler=None):
    model.train()
    metric_logger = new_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", new_utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(image)[0]  # ViT do tra 4-tuple -> lay logits
            loss = criterion(logits, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        acc1, _ = new_utils.accuracy(logits, target, topk=(1, min(3, logits.shape[1])))
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=image.shape[0])


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = new_utils.MetricLogger(delimiter="  ")
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, "Test:"):
            image, target = image.to(device), target.to(device)
            logits = model(image)[0]
            loss = criterion(logits, target)
            acc1, _ = new_utils.accuracy(logits, target, topk=(1, min(3, logits.shape[1])))
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=image.shape[0])
    metric_logger.synchronize_between_processes()
    print(f"Test Acc@1 {metric_logger.acc1.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def main(args):
    if args.output_dir:
        new_utils.mkdir(args.output_dir)
    new_utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    num_classes = len(dataset.classes)
    print("Classes:", dataset.classes)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True)

    model = build_teacher(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device)
        return

    print("Start teacher fine-tune")
    start = time.time()
    best = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, scaler)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device)
        if args.output_dir:
            ckpt = {"model": model_without_ddp.state_dict(), "epoch": epoch,
                    "classes": dataset.classes, "args": args}
            new_utils.save_on_master(ckpt, os.path.join(args.output_dir, "teacher_3cls.pth"))
            if acc > best:
                best = acc
                new_utils.save_on_master(ckpt, os.path.join(args.output_dir, "teacher_3cls_best.pth"))
    print(f"Teacher fine-tune xong sau {datetime.timedelta(seconds=int(time.time()-start))}, best acc@1={best:.3f}")


def get_args_parser(add_help=True):
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune ViT teacher xuong N lop", add_help=add_help)
    p.add_argument("--data-path", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("-b", "--batch-size", default=32, type=int)
    p.add_argument("--epochs", default=15, type=int)
    p.add_argument("-j", "--workers", default=2, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--wd", "--weight-decay", default=0.05, type=float, dest="weight_decay")
    p.add_argument("--label-smoothing", default=0.1, type=float)
    p.add_argument("--print-freq", default=10, type=int)
    p.add_argument("--output-dir", default="/kaggle/working", type=str)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--test-only", action="store_true")
    # cac co load_data cua dist_train_cakd can (dat mac dinh khop)
    p.add_argument("--interpolation", default="bilinear", type=str)
    p.add_argument("--val-resize-size", default=256, type=int)
    p.add_argument("--val-crop-size", default=224, type=int)
    p.add_argument("--train-crop-size", default=224, type=int)
    p.add_argument("--cache-dataset", action="store_true")
    p.add_argument("--auto-augment", default=None, type=str)
    p.add_argument("--random-erase", default=0.0, type=float)
    p.add_argument("--ra-magnitude", default=9, type=int)
    p.add_argument("--augmix-severity", default=3, type=int)
    p.add_argument("--ra-sampler", action="store_true")
    p.add_argument("--ra-reps", default=3, type=int)
    p.add_argument("--weights", default=None, type=str)
    p.add_argument("--world-size", default=1, type=int)
    p.add_argument("--dist-url", default="env://", type=str)
    return p


if __name__ == "__main__":
    main(get_args_parser().parse_args())
```

- [ ] **Step 2: Viết `experiments/run_teacher_kaggle.sh`**

```bash
# experiments/run_teacher_kaggle.sh
# GD1: fine-tune teacher ViT-B/16 -> 3 lop. Chay tu thu muc CAKD/.
torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 15 --lr 2e-4 --amp \
  --output-dir /kaggle/working
```

- [ ] **Step 3: Kiểm tra cú pháp Python (local, không GPU)**

Run: `python -c "import ast; ast.parse(open('CAKD/dist_train_teacher.py').read()); print('OK')"`
Expected: `OK` (không SyntaxError). Không import torch nên chạy được trên Windows.

- [ ] **Step 4: Smoke test trên Kaggle (sau khi có data + setup) — ghi vào KAGGLE_HUONGDAN.md ở Task 6**

Lệnh (chạy trên Kaggle, không phải local):
`torchrun --nproc_per_node=1 dist_train_teacher.py --data-path /kaggle/working/data_if --epochs 2 --batch-size 16 --amp --output-dir /kaggle/working`
Expected: in `Classes: ['glass', 'paper', 'plastic']`, loss giảm dần, cuối in `best acc@1=...`, có file `/kaggle/working/teacher_3cls.pth`.

- [ ] **Step 5: Commit**

```bash
git add CAKD/dist_train_teacher.py experiments/run_teacher_kaggle.sh
git commit -m "feat: script GD1 fine-tune teacher ViT xuong 3 lop"
```

---

## Task 4: Sửa `dist_train_cakd.py` — nạp teacher 3 lớp + student pretrained + cờ lịch distill

**Files:**
- Modify: `CAKD/dist_train_cakd.py` (phần `main` tạo model; hàm `get_args_parser`; công thức `λ` trong `train_one_epoch`)
- Create: `experiments/run_cakd_kaggle.sh`

**Interfaces:**
- Consumes: `teacher_3cls.pth` từ Task 3 (`ckpt["model"]`), `build_teacher` từ `dist_train_teacher.py`.
- Produces: `checkpoint.pth` (student ResNet-50 3 lớp) trong `--output-dir`.

- [ ] **Step 1: Thêm import `build_teacher` (đầu file, sau các import hiện có)**

Sửa `CAKD/dist_train_cakd.py`, ngay dưới dòng `from torchvision.models import (ViT_B_16_Weights,)`:

```python
from dist_train_teacher import build_teacher  # dung chung cach dung ViT 3 lop
```

- [ ] **Step 2: Thay đoạn tạo teacher & student trong `main`**

Tìm khối (khoảng dòng 410-419):

```python
    print("Creating model")
    # >>> TẠO BỘ BA MODEL CỦA CAKD <<<
    model = torchvision.models.resnet50_cakd(
        num_classes=num_classes
    )  # STUDENT (ResNet-50 độ thêm)
    teacher = torchvision.models.vit_b_16(
        weights=ViT_B_16_Weights.IMAGENET1K_V1
    )  # TEACHER (ViT pretrain)
```

Thay bằng:

```python
    print("Creating model")
    # >>> STUDENT: ResNet-50 pretrained ImageNet, roi thay fc -> num_classes <<<
    if args.student_pretrained:
        # weights ep num_classes=1000 va nap strict=False (bo qua pca/gl/cls_proj)
        model = torchvision.models.resnet50_cakd(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        model.fc = nn.Linear(512 * 4, num_classes)  # thay dau phan loai -> num_classes
        nn.init.zeros_(model.fc.bias)
    else:
        model = torchvision.models.resnet50_cakd(num_classes=num_classes)

    # >>> TEACHER: ViT-B/16 da fine-tune xuong num_classes lop (tu GD1) <<<
    teacher = build_teacher(num_classes, pretrained=False)  # khung 3 lop, chua trong so
    tea_ckpt = torch.load(args.teacher_weights, map_location="cpu")
    teacher.load_state_dict(tea_ckpt["model"] if "model" in tea_ckpt else tea_ckpt)
    print(f"Da nap teacher 3 lop tu {args.teacher_weights}")
```

- [ ] **Step 3: Nén lịch λ(epoch) bằng cờ mới trong `train_one_epoch`**

Tìm trong `train_one_epoch` (khoảng dòng 146):

```python
            loss = cls_loss + min(max(epoch - 25, 0) / 50.0, 0.2) * 1.0 * (
```

Thay bằng:

```python
            _lam = min(max(epoch - args.distill_start, 0) / float(args.distill_ramp), 0.2)
            loss = cls_loss + _lam * 1.0 * (
```

- [ ] **Step 4: Thêm 3 cờ mới vào `get_args_parser`**

Thêm ngay trước `return parser` (cuối hàm `get_args_parser`):

```python
    parser.add_argument(
        "--teacher-weights", default="/kaggle/working/teacher_3cls.pth", type=str,
        help="checkpoint teacher da fine-tune xuong so lop muc tieu (tu GD1)")
    parser.add_argument(
        "--student-pretrained", action="store_true",
        help="nap ResNet-50 pretrained ImageNet cho student roi thay fc")
    parser.add_argument(
        "--distill-start", default=5, type=int,
        help="epoch bat dau distill (goc ImageNet = 25)")
    parser.add_argument(
        "--distill-ramp", default=20, type=int,
        help="so epoch tang dan lambda len tran 0.2 (goc ImageNet = 50)")
```

- [ ] **Step 5: Kiểm tra cú pháp Python (local)**

Run: `python -c "import ast; ast.parse(open('CAKD/dist_train_cakd.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Viết `experiments/run_cakd_kaggle.sh`**

```bash
# experiments/run_cakd_kaggle.sh
# GD2: CAKD distill ViT(3 lop) -> ResNet-50(3 lop). Chay tu thu muc CAKD/.
torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-pretrained \
  --batch-size 32 --lr 0.01 --epochs 60 \
  --lr-warmup-epochs 5 --lr-warmup-method linear \
  --distill-start 5 --distill-ramp 20 \
  --auto-augment ta_wide --random-erase 0.1 --mixup-alpha 0.2 \
  --label-smoothing 0.1 --model-ema --amp \
  --train-crop-size 224 --val-resize-size 224 \
  --output-dir /kaggle/working/results
```

- [ ] **Step 7: Smoke test trên Kaggle (ghi vào KAGGLE_HUONGDAN.md) — 2 epoch**

Lệnh (Kaggle): như trên nhưng `--epochs 2 --batch-size 16 --distill-start 0 --distill-ramp 2`.
Expected: KHÔNG lỗi shape ở `gl_loss`; log in `pca_loss/gl_loss/cls_loss/gan_loss`; sinh `results/checkpoint.pth`.

- [ ] **Step 8: Commit**

```bash
git add CAKD/dist_train_cakd.py experiments/run_cakd_kaggle.sh
git commit -m "feat: CAKD nap teacher 3 lop + student pretrained + co lich distill"
```

---

## Task 5: Hướng dẫn Kaggle (B0→B4) + demo suy luận

**Files:**
- Create: `KAGGLE_HUONGDAN.md`

**Interfaces:**
- Consumes: tất cả script Task 1–4.
- Produces: tài liệu thứ tự các ô notebook để người dùng copy-paste chạy trên Kaggle.

- [ ] **Step 1: Viết `KAGGLE_HUONGDAN.md`**

````markdown
# Chạy CAKD 3 lớp trên Kaggle — thứ tự các ô notebook

> Bật GPU: Notebook → Settings → Accelerator → GPU T4/P100. Thêm dataset của bạn vào `/kaggle/input`.

## Ô 1 — Cài đặt & clone code
```bash
!pip install torch==1.12.0+cu113 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
!pip install einops
# Đưa code project vào Kaggle: upload repo thành dataset, hoặc git clone nếu có remote.
%cd /kaggle/working
!cp -r /kaggle/input/<ten-dataset-code>/CAKD /kaggle/working/CAKD
!cp -r /kaggle/input/<ten-dataset-code>/tools /kaggle/working/tools
```

## Ô 2 — Reorg data → ImageFolder
```bash
!python /kaggle/working/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/<ten-dataset-anh> \
  --dst /kaggle/working/data_if
```
Kiểm tra output in thống kê số ảnh mỗi lớp hợp lý.

## Ô 3 — Setup 3 file độ + verify (RESTART kernel sau ô 1 nếu vừa cài torch)
```bash
%cd /kaggle/working/CAKD
!python setup_kaggle.py
```
Phải thấy `[VERIFY] OK`. Nếu lỗi torch version → xem mục "Sự cố" cuối file.

## Ô 4 — GĐ1: fine-tune teacher
```bash
%cd /kaggle/working/CAKD
!torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 15 --lr 2e-4 --amp --output-dir /kaggle/working
```
Kết thúc có `/kaggle/working/teacher_3cls.pth` và in `best acc@1`.

## Ô 5 — GĐ2: CAKD distill
```bash
%cd /kaggle/working/CAKD
!bash /kaggle/working/CAKD/../experiments/run_cakd_kaggle.sh
```
(hoặc dán trực tiếp lệnh trong `experiments/run_cakd_kaggle.sh`.)

## Ô 6 — Đánh giá trên tập test
```bash
%cd /kaggle/working/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if_test \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --test-only --resume /kaggle/working/results/checkpoint.pth --batch-size 32
```
> Lưu ý: tạo `data_if_test` bằng cách reorg với `--splits test` rồi đổi tên `test`→`val`,
> hoặc trỏ `--data-path` sang thư mục có `val/` = tập test. Acc@5 vô nghĩa với 3 lớp.

## Ô 7 — Demo suy luận 1 ảnh
```python
import torch, torchvision
from PIL import Image
from new_utils import ClassificationPresetEval

classes = ["glass", "paper", "plastic"]
model = torchvision.models.resnet50_cakd(num_classes=3).cuda().eval()
ck = torch.load("/kaggle/working/results/checkpoint.pth", map_location="cpu")
model.load_state_dict(ck["model"])
tf = ClassificationPresetEval(crop_size=224, resize_size=224)
img = tf(Image.open("/kaggle/input/<...>/some.jpg").convert("RGB")).unsqueeze(0).cuda()
with torch.inference_mode():
    logits = model(img)[0]
    prob = logits.softmax(1)[0]
for c, p in sorted(zip(classes, prob.tolist()), key=lambda x: -x[1]):
    print(f"{c:8s} {p:.3f}")
```

## Sự cố thường gặp
- **`AttributeError: resnet50_cakd`** → chưa chạy ô 3 (chưa chép file độ) hoặc chưa restart kernel.
- **Lỗi shape ở gl_loss** → teacher chưa phải 3 lớp; kiểm tra `--teacher-weights` trỏ đúng `teacher_3cls.pth`.
- **Cài torch 1.12 xung đột** → thử `--force-reinstall`, hoặc dùng image Kaggle cũ hơn.
- **Hết VRAM** → giảm `--batch-size` (16/8), giữ `--amp`.
````

- [ ] **Step 2: Commit**

```bash
git add KAGGLE_HUONGDAN.md
git commit -m "docs: huong dan chay CAKD 3 lop tren Kaggle (B0-B4 + demo)"
```

---

## Self-Review Notes (đã kiểm)

- **Spec coverage:** B0(setup)=Task 2; B1(reorg)=Task 1; B2(teacher)=Task 3; B3(cakd sửa)=Task 4; B4(eval+demo)=Task 5. §5 hyperparams → nằm trong run scripts Task 3/4. ✔
- **Type consistency:** `build_teacher(num_classes, pretrained)` định nghĩa ở Task 3, dùng ở Task 4 ✔. `teacher_3cls.pth` lưu `{"model": state_dict}` ở Task 3, nạp `ckpt["model"]` ở Task 4 ✔. `reorg_to_imagefolder(...)` chữ ký khớp test ✔. `install_modified_files(...)` khớp test ✔.
- **Placeholder scan:** không có TBD/TODO; mọi bước có code/command cụ thể. `<ten-dataset>` trong hướng dẫn là chỗ người dùng điền đường dẫn Kaggle thực (không phải placeholder code).
- **Lưu ý thực thi:** Task 1–2 test được local (Windows, không GPU). Task 3–4 chỉ verify cú pháp local; smoke run thật trên Kaggle GPU (đã ghi lệnh + expected).
