"""Train MobileNetV3-Small BINH THUONG (khong teacher, khong distill) — baseline de so sanh voi CAKD.

Muc dich: cho thay KD co an thua khong. So MobileNetV3-Small tu train thuong (day) voi
MobileNetV3-Small hoc tu teacher ViT qua CAKD (o dist_train_cakd.py). Cung kien truc backbone,
chi khac co/khong distillation.

File TU CHUA: khong import gi tu repo, dan nguyen 1 file len Kaggle la chay.
Ho tro 2 kieu cau truc dataset (--layout auto tu do):
  - flat:         <data-path>/<class>/*.jpg          -> TU TACH train/val/test (10-class garbage,...)
  - split_images: <data-path>/<class>/<split>/images -> dung san split (15k-image-trash)
--classes bo trong = tu nhan tat ca thu muc con lam lop.

Chay tren Kaggle (notebook, GPU T4/P100):
    # dataset 10 lop kieu flat (vd garbage-classification):
    !python kaggle_train_mobilenetv3_baseline.py --data-path /kaggle/input/<ten>/<thu-muc-chua-lop>
    # tuy chinh:
    !python kaggle_train_mobilenetv3_baseline.py --data-path ... --epochs 30 --data-fraction 0.25

Ket qua trong --output-dir (mac dinh /kaggle/working/baseline_mobilenetv3):
    mobilenetv3_baseline_best.pth       trong so tot nhat theo val acc  <- LAY FILE NAY
    checkpoint_last.pth                 epoch cuoi
    history_mobilenetv3_baseline.json   lich su tung epoch, CUNG FORMAT voi history_cakd.json
                                        -> ve bieu do: python tools/plot_training.py --history <file>
Ket thuc script tu dong danh gia bo tot nhat tren tap test va in Acc@1.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================== DATASET =====================================
class ImgList(Dataset):
    """Dataset tu 1 danh sach (path, label). Dung chung cho moi layout."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _list_images(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]


def discover_classes(root):
    """Tu nhan cac lop = moi thu muc con cua root (theo thu tu alphabet)."""
    return sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))


def detect_layout(root, classes):
    """split_images = <root>/<class>/<split>/images/*  ;  flat = <root>/<class>/*.jpg"""
    if os.path.isdir(os.path.join(root, classes[0], "train", "images")):
        return "split_images"
    return "flat"


def gather_split_images(root, classes):
    """Layout co san train/val/test: <root>/<class>/<split>/images/*."""
    out = {"train": [], "val": [], "test": []}
    for idx, cls in enumerate(classes):
        for split in out:
            for p in _list_images(os.path.join(root, cls, split, "images")):
                out[split].append((p, idx))
    return out


def gather_flat(root, classes, val_frac, test_frac, seed=42):
    """Layout phang <root>/<class>/*.jpg -> TU TACH train/val/test (stratified, tat dinh)."""
    rng = np.random.RandomState(seed)
    out = {"train": [], "val": [], "test": []}
    for idx, cls in enumerate(classes):
        imgs = _list_images(os.path.join(root, cls))
        if not imgs:
            raise RuntimeError(f"Lop '{cls}' khong co anh nao trong {os.path.join(root, cls)}")
        imgs = np.array(imgs)
        rng.shuffle(imgs)
        n = len(imgs)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        parts = {"test": imgs[:n_test], "val": imgs[n_test:n_test + n_val],
                 "train": imgs[n_test + n_val:]}
        for split, arr in parts.items():
            for p in arr:
                out[split].append((str(p), idx))
    return out


def _subset_samples(samples, fraction, seed=42):
    """Lay 1 phan danh sach (path,label), CHIA DEU theo lop, tat dinh theo seed."""
    labels = np.asarray([lab for _, lab in samples])
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * fraction)))
        keep.extend(idx[:n].tolist())
    return [samples[i] for i in sorted(keep)]


def build_loaders(args):
    classes = args.classes or discover_classes(args.data_path)
    layout = args.layout if args.layout != "auto" else detect_layout(args.data_path, classes)
    print(f"Layout: {layout} | {len(classes)} lop: {classes}")
    if layout == "split_images":
        samples = gather_split_images(args.data_path, classes)
    else:
        samples = gather_flat(args.data_path, classes, args.val_split, args.test_split)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])
    loaders = {}
    for split, tf, shuffle in [("train", train_tf, True), ("val", eval_tf, False), ("test", eval_tf, False)]:
        s = samples[split]
        if split == "train" and args.data_fraction < 1.0:
            before = len(s)
            s = _subset_samples(s, args.data_fraction)
            print(f"  train: chi dung {args.data_fraction:.0%} ({before} -> {len(s)} anh)")
        loaders[split] = DataLoader(ImgList(s, tf), batch_size=args.batch_size, shuffle=shuffle,
                                    num_workers=args.workers, pin_memory=True)
        print(f"  {split:5s}: {len(s)} anh")
    return loaders, classes


# ============================== TRAIN / EVAL ================================
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """optimizer=None -> che do danh gia. Tra ve (loss trung binh, acc@1 %)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for image, target in loader:
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * image.size(0)
            correct += (output.argmax(1) == target).sum().item()
            seen += image.size(0)
    return total_loss / seen, 100.0 * correct / seen


def main():
    p = argparse.ArgumentParser(description="Baseline MobileNetV3-Small thuong (khong distill)")
    p.add_argument("--data-path", default="/kaggle/input/datasets/triuquct/15k-image-trash/15K_Image")
    p.add_argument("--classes", nargs="+", default=None,
                   help="danh sach lop; BO TRONG = tu nhan tat ca thu muc con cua data-path")
    p.add_argument("--layout", default="auto", choices=["auto", "flat", "split_images"],
                   help="flat = <class>/*.jpg (tu tach train/val/test); "
                        "split_images = <class>/<split>/images/*; auto = tu do")
    p.add_argument("--val-split", type=float, default=0.1, help="ti le val khi layout=flat")
    p.add_argument("--test-split", type=float, default=0.1, help="ti le test khi layout=flat")
    p.add_argument("--output-dir", default="/kaggle/working/baseline_mobilenetv3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01, help="SGD lr; giu nhu baseline khac de so sanh")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="chi dung 1 phan tap train (vd 0.25 = 1/4), chia deu theo lop")
    p.add_argument("--no-pretrained", action="store_true", help="train tu dau (mac dinh: nap ImageNet)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print("Nap du lieu...")
    loaders, classes = build_loaders(args)

    # MobileNetV3-Small chuan torchvision (~2.5M tham so), thay lop cuoi ra so lop bai toan.
    weights = None if args.no_pretrained else MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    history, best_acc, best_epoch = [], 0.0, -1
    best_path = os.path.join(args.output_dir, "mobilenetv3_baseline_best.pth")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Ghi lich su CUNG FORMAT voi history_cakd.json -> tools/plot_training.py ve duoc
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc1": round(train_acc, 5),
            "test_acc1": round(val_acc, 5),
            "cls_loss": None, "pca_loss": None, "gl_loss": None, "gan_loss": None,
            "lr": round(lr_now, 8),
        })
        with open(os.path.join(args.output_dir, "history_mobilenetv3_baseline.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        flag = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "classes": classes,
                        "epoch": epoch, "val_acc1": val_acc}, best_path)
            flag = "  <- best, da luu"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "classes": classes},
                   os.path.join(args.output_dir, "checkpoint_last.pth"))

        print(f"Epoch {epoch:3d}/{args.epochs}  lr {lr_now:.5f}  "
              f"train loss {train_loss:.4f} acc {train_acc:.2f}%  |  val acc {val_acc:.2f}%"
              f"  ({time.time() - t0:.0f}s){flag}")

    print(f"\nBest val acc@1 = {best_acc:.2f}% @ epoch {best_epoch} -> {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_loss, test_acc = run_epoch(model, loaders["test"], criterion, device)
    print(f"TEST acc@1 (bo tot nhat) = {test_acc:.2f}%")


if __name__ == "__main__":
    main()
