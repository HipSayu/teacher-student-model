# =============================================================================
# mobilenet_cakd.py — STUDENT MobileNetV3-Small cho CAKD (student SIEU NHE).
# -----------------------------------------------------------------------------
# Y tuong: khung CAKD can 1 feature map 14x14 (=196 token) de cam 2 dau distill
#   - pca_proj (Attention)  -> sinh attention map giong ViT (196x196)
#   - gl_proj  (GLProj)     -> chieu feature CNN sang chieu ViT (196x768)
# MobileNetV3-Small: features[0:9] cho ra 48 kenh @14x14 (dung lam tin hieu distill),
# features[9:] + classifier lo phan phan loai. Tai dung Attention/GLProj tu resnet_cakd.
# Forward tra ve 4 thu GIONG resnet_cakd: (logits, [attn_qk, attn_vv], vit_feat, cls_token).
# =============================================================================
import torch
import torch.nn as nn
from torch import Tensor

from models.resnet_cakd import Attention, GLProj

# Chi so tach backbone MobileNetV3-Small (input 224):
#   features[0..8] -> 48 kenh @14x14 (196 token) ; features[9..12] -> 576 kenh @7x7
_SPLIT = 9
_MID_DIM = 48   # so kenh sau features[8]


class MobileNetV3Small_CAKD(nn.Module):
    def __init__(self, num_classes=1000, tgt_dim=768, num_patch=196, pretrained=False):
        super().__init__()
        from torchvision.models import (mobilenet_v3_small,
                                         MobileNet_V3_Small_Weights)
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        feats = base.features

        # Backbone tach lam 2 doan tai tang 14x14
        self.stem = nn.Sequential(*[feats[i] for i in range(_SPLIT)])          # -> 48 @14x14
        self.late = nn.Sequential(*[feats[i] for i in range(_SPLIT, len(feats))])  # -> 576 @7x7
        self.avgpool = base.avgpool                                            # AdaptiveAvgPool2d(1)
        last_dim = base.classifier[0].in_features   # 576
        hidden = base.classifier[0].out_features    # 1024

        # >>> 2 dau distill CAKD dat sau stem (14x14) <<<
        self.pca_proj = Attention(dim=_MID_DIM, heads=8, dim_head=int(_MID_DIM / 8))
        self.gl_proj = GLProj(src_dim=_MID_DIM, tgt_dim=tgt_dim, num_patch=num_patch)

        # Dau phan loai: giu nguyen kieu head cua MobileNetV3 (Linear->HSwish->Dropout->Linear)
        self.classifier = nn.Sequential(
            nn.Linear(last_dim, hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden, num_classes),
        )
        if pretrained:
            # nap lai Linear(576->1024) pretrained; lop cuoi (->num_classes) giu khoi tao
            self.classifier[0].load_state_dict(base.classifier[0].state_dict())

        # Chieu token CNN -> chieu ViT (khop cls distill)
        self.cls_proj = nn.Linear(last_dim, tgt_dim)

    def forward(self, x: Tensor):
        x = self.stem(x)                                  # (N,48,14,14)
        # feature map -> chuoi 196 token
        tmp = torch.reshape(x, (x.shape[0], x.shape[1], -1)).permute((0, 2, 1))  # (N,196,48)
        _, attn_qk, attn_vv = self.pca_proj(tmp)
        num_heads = attn_qk.shape[1]
        attn_qk = attn_qk.sum(dim=1) / num_heads          # trung binh head -> (N,196,196)
        attn_vv = attn_vv.sum(dim=1) / num_heads
        vit_feat = self.gl_proj(tmp)                      # (N,196,768)
        # nhanh phan loai
        x = self.late(x)                                  # (N,576,7,7)
        x = self.avgpool(x)
        cnn_token = torch.flatten(x, 1)                   # (N,576)
        logits = self.classifier(cnn_token)
        return logits, [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)


def mobilenetv3_small_cakd(num_classes=1000, pretrained=False):
    """Tao MobileNetV3Small_CAKD — student sieu nhe (~2.5M backbone). pretrained=True ->
    nap backbone MobileNetV3-Small ImageNet (lop phan loai cuoi va cac dau CAKD giu khoi tao)."""
    return MobileNetV3Small_CAKD(num_classes=num_classes, pretrained=pretrained)
