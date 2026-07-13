# =============================================================================
# vit_cakd.py — TEACHER ViT-B/16 (ban PORT sang torch 2.x, chay native).
# -----------------------------------------------------------------------------
# Thay vi chep de vision_transformer.py + functional.py cua torch 1.12, o day ta
# DUNG THANG torchvision.models.vit_b_16(weights=...) that (trong so dung, torch2-native),
# roi tu chay encoder de moi ra attention 2 lop cuoi.
#
# Attention khop dung ban sua functional.py cu:
#   attn_qk = q·k^T / sqrt(head_dim)   (raw, PRE-softmax; q da chia sqrt(head_dim))
#   attn_vv = v·v^T                    (raw, KHONG chia)
#   ca hai average theo head -> (N, 197, 197)
# Output logits TRUNG KHIT torchvision vit(x) (cung phep tinh) — da kiem chung bang test.
#
# Forward tra ve 4 thu: (logits, [qk2, vv2, qk1, vv1], cls_token, feats)
#   trong do [2]=qk1, [3]=vv1 la 2 map lop CUOI (student khop qua pca_loss + GAN).
# =============================================================================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ViT_B_16_Weights


def _process_input(vit, x):
    """Anh (N,3,224,224) -> chuoi 196 token (N,196,768). Sao y VisionTransformer._process_input."""
    n, c, h, w = x.shape
    p = vit.patch_size
    n_h = h // p
    n_w = w // p
    x = vit.conv_proj(x)
    x = x.reshape(n, vit.hidden_dim, n_h * n_w)
    x = x.permute(0, 2, 1)
    return x


def _block_forward_with_attn(block, x_in):
    """Chay 1 EncoderBlock GIONG torchvision nhung tra ve them (attn_qk, attn_vv) avg-head."""
    mha = block.self_attention
    num_heads = mha.num_heads
    x = block.ln_1(x_in)
    N, L, E = x.shape
    hd = E // num_heads
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)   # (N,L,3E)
    q, k, v = qkv.chunk(3, dim=-1)
    def split(t):
        return t.view(N, L, num_heads, hd).permute(0, 2, 1, 3)  # (N,H,L,hd)
    q, k, v = split(q), split(k), split(v)
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)  # (N,H,L,L) raw qk
    attn_vv = torch.matmul(v, v.transpose(-2, -1))               # (N,H,L,L) raw vv
    out = torch.matmul(F.softmax(attn, dim=-1), v)               # (N,H,L,hd)
    out = out.permute(0, 2, 1, 3).reshape(N, L, E)
    out = mha.out_proj(out)
    # phan con lai cua EncoderBlock: dropout -> residual -> ln_2 -> mlp -> residual
    x = block.dropout(out)
    x = x + x_in
    y = block.ln_2(x)
    y = block.mlp(y)
    return x + y, attn.mean(dim=1), attn_vv.mean(dim=1)


class TeacherCAKD(nn.Module):
    """ViT-B/16 teacher tra ve 4-tuple (logits, [4 attn], cls_token, feats) — torch2 native."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = torchvision.models.vit_b_16(weights=weights)
        self.vit.heads.head = nn.Linear(self.vit.hidden_dim, num_classes)
        nn.init.zeros_(self.vit.heads.head.bias)

    def forward(self, x):
        vit = self.vit
        x = _process_input(vit, x)
        n = x.shape[0]
        batch_class_token = vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # Encoder: input + pos_embedding -> dropout -> layers -> ln  (sao y torchvision)
        x = x + vit.encoder.pos_embedding
        x = vit.encoder.dropout(x)
        layers = vit.encoder.layers
        num_layers = len(layers)
        qk2 = vv2 = qk1 = vv1 = None
        for i in range(num_layers):
            if i < num_layers - 2:
                x = layers[i](x)
            elif i == num_layers - 2:
                x, qk2, vv2 = _block_forward_with_attn(layers[i], x)
            else:
                x, qk1, vv1 = _block_forward_with_attn(layers[i], x)
        x = vit.encoder.ln(x)
        cls_token = x[:, 0]
        feats = x[:, 1:]
        logits = vit.heads(cls_token)
        # thu tu khop ban goc: [lop ap chot qk, vv, lop cuoi qk, vv]
        return logits, [qk2, vv2, qk1, vv1], cls_token, feats


def build_teacher(num_classes, pretrained=True):
    return TeacherCAKD(num_classes, pretrained)
