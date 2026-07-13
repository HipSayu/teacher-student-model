# =============================================================================
# inference.py — Nap student ResNet-50 CAKD (model_60.pth) va suy luan 1 anh.
# -----------------------------------------------------------------------------
# model_60.pth la checkpoint tot nhat (epoch 60, test acc@1 = 96.73%) cua student
# ResNet-50 chung cat tri thuc tu teacher ViT-B/16 (CAKD), phan loai 3 lop:
#   glass / paper / plastic  (thu tu alphabet cua ImageFolder).
#
# Checkpoint la dict: {"model": state_dict, "optimizer": ..., "epoch": 60, ...}.
# Suy luan chi can key "model" -> nap vao resnet50_cakd(num_classes=3),
# lay dau ra thu 0 (logits) -> softmax.
# =============================================================================
import os
import sys
import time
from typing import List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- Dinh vi dinh nghia model (CAKD/models/resnet_cakd.py) ---------------------
# Cho phep chay du repo dat o dau: uu tien bien moi truong CAKD_DIR, sau do doan
# theo vi tri file nay (deploy/ nam canh CAKD/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("CAKD_DIR"),
    os.path.join(_HERE, "CAKD"),           # khi copy CAKD/ vao image Docker
    os.path.join(_HERE, "..", "CAKD"),     # khi chay tu repo goc
]
for _c in _CANDIDATES:
    if _c and os.path.isdir(os.path.join(_c, "models")):
        sys.path.insert(0, os.path.abspath(_c))
        break

from models.resnet_cakd import resnet50_cakd  # noqa: E402

# Thu tu lop PHAI trung voi ImageFolder (alphabet). Doi qua bien CLASS_NAMES neu can.
CLASS_NAMES: List[str] = os.environ.get(
    "CLASS_NAMES", "glass,paper,plastic"
).split(",")

# Tien xu ly GIONG HET luc danh gia (val-resize-size=224, val-crop-size=224 + chuan ImageNet).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)

# Cho /detect: ep ca khung ve 224x224 (KHONG crop) -> CAM phu TOAN BO khung ->
# bounding box normalize [0,1] khop truc tiep len preview cua app.
_TRANSFORM_DETECT = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


class TrashClassifier:
    """Bao boc model: nap 1 lan, tai su dung cho moi request."""

    def __init__(self, weights_path: str, device: str = None, use_ema: bool = False):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classes = CLASS_NAMES
        self.model = resnet50_cakd(num_classes=len(self.classes), pretrained=False)

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = self._extract_state_dict(ckpt, use_ema)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[inference] canh bao — thieu key: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        if unexpected:
            print(f"[inference] canh bao — thua key: {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")

        self.model.to(self.device).eval()
        self.epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None

        # Hook lay feature map cuoi (layer4 output, N x 2048 x 7 x 7) de tinh CAM cho /detect.
        self._feat = None

        def _grab_feat(_m, _in, out):
            self._feat = out.detach()

        self.model.layer4.register_forward_hook(_grab_feat)
        print(f"[inference] Da nap model tu {weights_path} | device={self.device} | classes={self.classes}")

    @staticmethod
    def _extract_state_dict(ckpt, use_ema: bool) -> Dict[str, torch.Tensor]:
        """Rut state_dict student tu checkpoint (ho tro ca ban EMA)."""
        if not isinstance(ckpt, dict):
            return ckpt
        if use_ema and "model_ema" in ckpt:
            raw = ckpt["model_ema"]
            # ExponentialMovingAverage boc module + co buffer 'n_averaged' -> bo tien to 'module.'
            return {
                k.replace("module.", "", 1): v
                for k, v in raw.items()
                if k != "n_averaged"
            }
        if "model" in ckpt:
            return ckpt["model"]
        return ckpt  # truong hop file chi chua state_dict tho

    @torch.inference_mode()
    def predict(self, image: Image.Image, topk: int = 3) -> Dict:
        """Suy luan 1 anh PIL -> nhan + xac suat softmax cho tung lop."""
        t0 = time.time()
        x = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]                 # dau ra thu 0 = logits
        probs = F.softmax(logits, dim=1)[0]
        infer_ms = (time.time() - t0) * 1000.0

        pairs = sorted(
            zip(self.classes, probs.tolist()), key=lambda p: p[1], reverse=True
        )
        topk = min(topk, len(pairs))
        return {
            "label": pairs[0][0],
            "confidence": round(pairs[0][1], 4),
            "topk": [{"label": c, "prob": round(p, 4)} for c, p in pairs[:topk]],
            "probs": {c: round(p, 4) for c, p in zip(self.classes, probs.tolist())},
            "inference_ms": round(infer_ms, 1),
        }

    @torch.inference_mode()
    def detect(self, image: Image.Image, thresh: float = 0.35) -> Dict:
        """Phan loai + dinh vi (weakly-supervised) bang CAM.

        Tra ve 1 bounding box (normalize [0,1] theo TOAN KHUNG) quanh vung model
        chu y nhat cho lop du doan. KHONG phai detection da vat — chi 1 box/vat chinh.
        """
        t0 = time.time()
        x = _TRANSFORM_DETECT(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]                 # kich hoat hook -> self._feat
        probs = F.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

        # CAM_c = sum_k w[c,k] * feature_map[k] (khong can gradient — dung thang trong so fc)
        feat = self._feat[0]                      # (2048, 7, 7)
        w = self.model.fc.weight[pred]            # (2048,)
        cam = torch.einsum("c,chw->hw", w, feat)  # (7, 7)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam[None, None], size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam - cam.min()
        maxv = cam.max()
        if maxv > 0:
            cam = cam / maxv

        mask = cam >= thresh
        ys, xs = torch.where(mask)
        if xs.numel() == 0:
            box = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
        else:
            x0, x1 = xs.min().item() / 224.0, (xs.max().item() + 1) / 224.0
            y0, y1 = ys.min().item() / 224.0, (ys.max().item() + 1) / 224.0
            box = {
                "x": round(x0, 4),
                "y": round(y0, 4),
                "w": round(x1 - x0, 4),
                "h": round(y1 - y0, 4),
            }

        infer_ms = (time.time() - t0) * 1000.0
        return {
            "label": self.classes[pred],
            "confidence": round(conf.item(), 4),
            "box": box,                           # {x, y, w, h} normalize [0,1] theo khung
            "probs": {c: round(p, 4) for c, p in zip(self.classes, probs.tolist())},
            "inference_ms": round(infer_ms, 1),
        }
