"""Train ResNet-18 BINH THUONG (khong teacher, khong distill) — baseline nho de so sanh voi ResNet-50 va CAKD.

File TU CHUA: khong import gi tu repo, dan nguyen 1 file len Kaggle la chay.
Doc thang cau truc dataset goc <data-path>/<class>/<split>/images/*.jpg
(glass/paper/plastic -> train/val/test) — KHONG can buoc reorg.

Chay tren Kaggle (notebook, GPU T4/P100):
    !python kaggle_train_resnet18_baseline.py
hoac tuy chinh:
    !python kaggle_train_resnet18_baseline.py --epochs 30 --batch-size 32 --lr 0.01

Ket qua trong --output-dir (mac dinh /kaggle/working/baseline_resnet18):
    resnet18_baseline_best.pth       trong so tot nhat theo val acc  <- LAY FILE NAY DE DUNG
    checkpoint_last.pth              epoch cuoi (de resume/doi chieu)
    history_resnet18_baseline.json   lich su tung epoch, CUNG FORMAT voi history_baseline.json
                                     -> ve so sanh: python tools/plot_report.py history_baseline.json history_resnet18_baseline.json
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
from torchvision.models import ResNet18_Weights, resnet18

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================== DATASET =====================================
class TrashDataset(Dataset):
    """Doc <root>/<class>/<split>/images/*.jpg. Nhan = chi so lop (theo thu tu --classes)."""

    def __init__(self, root, split, classes, transform):
        self.transform = transform
        self.samples = []
        for idx, cls in enumerate(classes):
            img_dir = os.path.join(root, cls, split, "images")
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"Khong thay {img_dir} — kiem tra --data-path/--classes")
            for f in sorted(os.listdir(img_dir)):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    self.samples.append((os.path.join(img_dir, f), idx))
        if not self.samples:
            raise RuntimeError(f"Khong co anh nao trong split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _subset_by_fraction(ds, fraction, seed=42):
    """Lay 1 phan tap train, CHIA DEU theo lop (stratified), tat dinh theo seed."""
    targets = np.asarray([lab for _, lab in ds.samples])
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * fraction)))
        keep.extend(idx[:n].tolist())
    ds.samples = [ds.samples[i] for i in sorted(keep)]
    return ds


def build_loaders(args):
    # Augment chuan cho finetune ImageNet-pretrained (muc "binh thuong", khong ta_wide/mixup)
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
        ds = TrashDataset(args.data_path, split, args.classes, tf)
        if split == "train" and args.data_fraction < 1.0:
            before = len(ds)
            ds = _subset_by_fraction(ds, args.data_fraction)
            print(f"  train: chi dung {args.data_fraction:.0%} ({before} -> {len(ds)} anh)")
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                    num_workers=args.workers, pin_memory=True)
        print(f"  {split:5s}: {len(ds)} anh")
    return loaders


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
    p = argparse.ArgumentParser(description="Baseline ResNet-18 thuong (khong distill)")
    p.add_argument("--data-path", default="/kaggle/input/datasets/triuquct/15k-image-trash/15K_Image")
    p.add_argument("--classes", nargs="+", default=["glass", "paper", "plastic"])
    p.add_argument("--output-dir", default="/kaggle/working/baseline_resnet18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01, help="SGD lr; giu nguyen nhu baseline ResNet-50 de so sanh")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="chi dung 1 phan tap train (vd 0.1 = 10%), chia deu theo lop")
    p.add_argument("--no-pretrained", action="store_true", help="train tu dau (mac dinh: nap ImageNet)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device} | Classes: {args.classes}")
    print("Nap du lieu...")
    loaders = build_loaders(args)

    # ResNet-18 chuan torchvision (~11.7M tham so, so voi 25.6M cua ResNet-50),
    # thay lop cuoi ra so lop cua bai toan. ResNet-18 chi co weights V1.
    weights = None if args.no_pretrained else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(args.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    history, best_acc, best_epoch = [], 0.0, -1
    best_path = os.path.join(args.output_dir, "resnet18_baseline_best.pth")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Ghi lich su CUNG FORMAT voi history_baseline.json de tools/plot_report.py so sanh duoc
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc1": round(train_acc, 5),
            "test_acc1": round(val_acc, 5),
            "cls_loss": None, "pca_loss": None, "gl_loss": None, "gan_loss": None,
            "lr": round(lr_now, 8),
        })
        with open(os.path.join(args.output_dir, "history_resnet18_baseline.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Luu best theo val acc (khong mat epoch dinh) + checkpoint cuoi
        flag = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "classes": args.classes,
                        "epoch": epoch, "val_acc1": val_acc}, best_path)
            flag = "  <- best, da luu"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "classes": args.classes},
                   os.path.join(args.output_dir, "checkpoint_last.pth"))

        print(f"Epoch {epoch:3d}/{args.epochs}  lr {lr_now:.5f}  "
              f"train loss {train_loss:.4f} acc {train_acc:.2f}%  |  val acc {val_acc:.2f}%"
              f"  ({time.time() - t0:.0f}s){flag}")

    # Danh gia bo TOT NHAT tren tap test (chi cham 1 lan, sau khi chon xong model)
    print(f"\nBest val acc@1 = {best_acc:.2f}% @ epoch {best_epoch} -> {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_loss, test_acc = run_epoch(model, loaders["test"], criterion, device)
    print(f"TEST acc@1 (bo tot nhat) = {test_acc:.2f}%")


if __name__ == "__main__":
    main()
