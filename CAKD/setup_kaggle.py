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
    tv_models = os.path.dirname(torchvision.models.__file__)
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
    print("[VERIFY] OK - student & teacher tra ve 4-tuple dung shape (3 lop).")


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
