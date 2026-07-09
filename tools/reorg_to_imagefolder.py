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
