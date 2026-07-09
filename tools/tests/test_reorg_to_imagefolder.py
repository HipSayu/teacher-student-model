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
