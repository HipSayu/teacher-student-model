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
