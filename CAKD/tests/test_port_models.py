import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.resnet_cakd import resnet50_cakd
from models.vit_cakd import build_teacher


def test_student_returns_4tuple_shapes():
    student = resnet50_cakd(num_classes=3, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        logits, attn, feat, token = student(x)
    assert logits.shape == (2, 3)
    assert attn[0].shape == (2, 196, 196)   # attn_qk
    assert attn[1].shape == (2, 196, 196)   # attn_vv
    assert feat.shape == (2, 196, 768)      # vit_feat
    assert token.shape == (2, 768)          # cls_proj token


def test_teacher_returns_4tuple_shapes():
    teacher = build_teacher(3, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        logits, attn, cls_token, feats = teacher(x)
    assert logits.shape == (2, 3)
    assert len(attn) == 4
    for a in attn:
        assert a.shape == (2, 197, 197)
    assert cls_token.shape == (2, 768)
    assert feats.shape == (2, 196, 768)


def test_teacher_logits_match_torchvision_vit():
    """Logits cua TeacherCAKD phai trung khit self.vit(x) (cung phep tinh)."""
    teacher = build_teacher(3, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        ours = teacher(x)[0]
        ref = teacher.vit(x)
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max().item()


def test_cakd_loss_shapes_align():
    """Kiem tra shape khop giua student va teacher o cac diem tinh loss CAKD."""
    student = resnet50_cakd(num_classes=3, pretrained=False).eval()
    teacher = build_teacher(3, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        s_logits, s_attn, s_feat, s_token = student(x)
        t_logits, t_attn, t_token, t_feat = teacher(x)
    # gl_loss: logits 3 vs 3 (khong con vo shape 3 vs 1000)
    assert s_logits.shape == t_logits.shape == (2, 3)
    # gl_loss: token & feat
    assert s_token.shape == t_token.shape == (2, 768)
    assert s_feat.shape == t_feat.shape == (2, 196, 768)
    # pca_loss & GAN: attention (bo class token cua teacher)
    assert s_attn[0].shape == t_attn[2][:, 1:, 1:].shape == (2, 196, 196)
    assert s_attn[1].shape == t_attn[3][:, 1:, 1:].shape == (2, 196, 196)
