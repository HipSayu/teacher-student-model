# =============================================================================
# resnet.py — STUDENT MODEL của hệ thống CAKD (bản ResNet đã được "độ" thêm)
# -----------------------------------------------------------------------------
# Đây là bản chép từ torchvision/models/resnet.py rồi SỬA để phục vụ CAKD.
# File gốc chỉ định nghĩa ResNet thường (phân loại ảnh). Bản này THÊM:
#   - Các khối kiểu Transformer: PreNorm, FeedForward, Attention, GLProj
#   - Class ResNet_CAKD: ResNet nhưng ở forward TRẢ VỀ THÊM attention & feature
#     để so khớp (distill) với teacher là Vision Transformer (ViT).
#
# Ý tưởng CAKD (Cross-Architecture KD): CNN (ResNet) và ViT có kiến trúc KHÁC
# nhau -> không thể copy thẳng đặc trưng. Nên student CNN được gắn thêm vài
# lớp "phiên dịch" (pca_proj, gl_proj, cls_proj) để tạo ra thứ CÓ DẠNG giống
# đầu ra của ViT (attention map + token feature), rồi ép chúng giống teacher.
#
# CHIA FILE THÀNH 4 PHẦN:
#   PHẦN A (dòng ~44-66):   Bảng chỉ số idx_196/idx_49 — ánh xạ lưới CNN <-> patch ViT
#   PHẦN B (dòng ~69-152):  Khối kiểu Transformer: PreNorm, FeedForward, Attention, GLProj
#   PHẦN C (dòng ~155-400): Xương sống ResNet chuẩn: conv, BasicBlock, Bottleneck, ResNet
#   PHẦN D (dòng ~420-571): ResNet_CAKD — bản độ thêm nhánh distill (QUAN TRỌNG NHẤT)
#   PHẦN E (dòng ~574-cuối):Metadata trọng số pretrain + hàm factory (boilerplate torchvision)
# =============================================================================

from functools import partial   # partial: gói sẵn 1 hàm kèm tham số (dùng cho transforms)
from typing import Type, Any, Callable, Union, List, Optional   # type hints, không ảnh hưởng chạy
from collections import OrderedDict   # dict giữ đúng thứ tự (dùng cho các lớp Linear trong GLProj)

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat   # einops: đổi shape tensor bằng cú pháp "b n (h d) -> b h n d" cho dễ đọc
from einops.layers.torch import Rearrange

# ----- Các import nội bộ của torchvision (đường dẫn tương đối ".." nên file này
#       PHẢI đặt vào trong cây torchvision/models mới chạy được) -----
from ..transforms._presets import ImageClassification   # pipeline tiền xử lý ảnh chuẩn
from ..utils import _log_api_usage_once                 # ghi log thống kê dùng API
from ._api import WeightsEnum, Weights                  # kiểu liệt kê bộ trọng số pretrain
from ._meta import _IMAGENET_CATEGORIES                 # danh sách 1000 lớp ImageNet
from ._utils import handle_legacy_interface, _ovewrite_named_param   # tiện ích tương thích API cũ


# __all__: danh sách tên được "công khai" khi ai đó gọi `from resnet import *`.
# Chú ý có thêm "resnet18_cakd" và "resnet50_cakd" — hai hàm tạo student cho CAKD.
__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet18_cakd",
    "resnet34",
    "resnet50",
    "resnet50_cakd",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

# =============================================================================
# PHẦN A — BẢNG CHỈ SỐ ÁNH XẠ LƯỚI CNN <-> PATCH CỦA ViT
# -----------------------------------------------------------------------------
# Vấn đề: teacher ViT chia ảnh 224x224 thành lưới 14x14 = 196 patch. Student CNN
# ở layer3 cho ra feature map cỡ 14x14 = 196 ô -> khớp 1-1 với 196 patch ViT.
# Nhưng độ phân giải KHÔNG phải lúc nào cũng trùng: ví dụ khi so 196 ô CNN với
# một cách gom nhóm khác, cần biết "ô CNN nào thuộc về nhóm nào".
#
# Mỗi list idx_224_16_k liệt kê CHỈ SỐ các ô (trong 196 ô đã duỗi phẳng) cùng
# thuộc về nhóm/patch thứ k. GLProj dùng bảng này để đưa từng nhóm ô qua một
# lớp Linear riêng -> "phiên dịch" đặc trưng CNN sang không gian đặc trưng ViT.
#   idx_196: gom 196 ô thành 16 nhóm (dùng khi num_patch=196)
#   idx_49 : gom 49 ô  thành 4  nhóm (dùng khi num_patch=49, ảnh/độ phân giải nhỏ hơn)
# (Các con số này là bảng tra cứng — được tính sẵn cho đúng cách sắp xếp patch.)
# =============================================================================
idx_224_16_0 = [0,1,2,3,14,15,16,17,28,29,30,31,42,43,44,45]
idx_224_16_1 = [4,5,6,7,18,19,20,21,32,33,34,35,46,47,48,49]
idx_224_16_2 = [8,9,10,11,22,23,24,25,36,37,38,39,50,51,52,53]
idx_224_16_3 = [12,13,26,27,40,41,54,55]
idx_224_16_4 = [ 56,  57,  58,  59,  70,  71,  72,  73,  84,  85,  86,  87,  98,  99, 100, 101]
idx_224_16_5 = [ 60,  61,  62,  63,  74,  75,  76,  77,  88,  89,  90,  91,  102, 103, 104, 105]
idx_224_16_6 = [ 64,  65,  66,  67,  78,  79,  80,  81,  92,  93,  94,  95, 106,  107, 108, 109]
idx_224_16_7 = [ 68,  69,  82,  83,  96,  97, 110,  111]
idx_224_16_8 = [112, 113, 114, 115, 126, 127, 128, 129, 140, 141, 142, 143, 154,  155, 156, 157]
idx_224_16_9 = [116, 117, 118, 119, 130, 131, 132, 133, 144, 145, 146, 147, 158,  159, 160, 161]
idx_224_16_10 = [120, 121, 122, 123, 134, 135, 136, 137, 148, 149, 150, 151, 162,  163, 164, 165]
idx_224_16_11 = [124,125,138,139,152,153,166,167]
idx_224_16_12 = [168,169,170,171,182,183,184,185]
idx_224_16_13 = [172,173,174,175,186,187,188,189]
idx_224_16_14 = [176,177,178,179,190,191,192,193]
idx_224_16_15 = [180,181,194,195]
idx_196 = [idx_224_16_0,idx_224_16_1,idx_224_16_2,idx_224_16_3,idx_224_16_4,idx_224_16_5,idx_224_16_6,idx_224_16_7,idx_224_16_8,idx_224_16_9,idx_224_16_10,idx_224_16_11,idx_224_16_12,idx_224_16_13,idx_224_16_14,idx_224_16_15]

idx_224_32_0 = [0,1,2,3,7,8,9,10,14,15,16,17,21,22,23,24]
idx_224_32_1 = [4,5,6,11,12,13,18,19,20,25,26,27]
idx_224_32_2 = [28,29,30,31,35,36,37,38,42,43,44,45]
idx_224_32_3 = [32,33,34,39,40,41,46,47,48]
idx_49 = [idx_224_32_0,idx_224_32_1,idx_224_32_2,idx_224_32_3]


# =============================================================================
# PHẦN B — CÁC KHỐI KIỂU TRANSFORMER (gắn vào CNN để bắt chước ViT)
# =============================================================================

class PreNorm(nn.Module):
    """Chuẩn hóa TRƯỚC rồi mới đưa qua hàm fn (kiểu "pre-norm" của Transformer).
    Giúp train sâu ổn định hơn. (Định nghĩa sẵn ở đây nhưng CAKD chủ yếu xài Attention.)"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)   # chuẩn hóa theo chiều đặc trưng (LayerNorm, không phải BatchNorm)
        self.fn = fn                    # hàm/khối sẽ chạy sau khi chuẩn hóa (vd Attention, FeedForward)
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)   # x -> norm -> fn

class FeedForward(nn.Module):
    """Khối MLP 2 lớp của Transformer: Linear -> GELU -> Linear (kèm dropout)."""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),   # nở rộng chiều đặc trưng
            nn.GELU(),                    # kích hoạt phi tuyến (mượt hơn ReLU, chuẩn của Transformer)
            nn.Dropout(dropout),          # tắt ngẫu nhiên vài nơ-ron (chống overfit)
            nn.Linear(hidden_dim, dim),   # thu về chiều ban đầu
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Self-Attention đa đầu (multi-head) — TRÁI TIM của Transformer.
    Trong CAKD, khối này (đặt tên pca_proj) được gắn lên feature map CNN để SINH RA
    một "attention map" có dạng giống attention của teacher ViT, phục vụ distill.
    Điểm đặc biệt: forward trả về CẢ ma trận attention thô (dots_qk, dots_vv) chứ
    không chỉ output — vì CAKD cần chính các ma trận này để so khớp với teacher.
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads     # tổng chiều sau khi ghép tất cả các đầu (head)
        project_out = not (heads == 1 and dim_head == dim)   # có cần lớp chiếu đầu ra không

        self.heads = heads                # số "đầu" attention (nhìn dữ liệu theo nhiều góc)
        self.scale = dim_head ** -0.5     # hệ số chia 1/sqrt(d) để ổn định softmax (chuẩn Transformer)

        self.attend = nn.Softmax(dim = -1)   # softmax biến điểm tương quan thành trọng số (tổng =1)
        self.dropout = nn.Dropout(dropout)

        # to_qkv: từ 1 đầu vào sinh RA CẢ Q, K, V cùng lúc (nhân 3) trong 1 phép Linear
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # to_out: gộp các đầu lại và chiếu về chiều dim ban đầu (nếu cần)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()   # nếu không cần thì để nguyên (Identity)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)   # tính rồi cắt làm 3 phần: q, k, v
        # Tách mỗi phần thành nhiều đầu: gộp (h d) -> tách riêng chiều head h và chiều d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # dots_qk: độ tương quan giữa mỗi vị trí (query) với mọi vị trí (key) -> "ai chú ý tới ai"
        dots_qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # dots_vv: tương quan giữa các value với nhau (biến thể riêng của CAKD, dùng làm tín hiệu distill thứ 2)
        dots_vv = torch.matmul(v, v.transpose(-1, -2)) * self.scale

        attn_qk = self.attend(dots_qk)    # softmax -> trọng số attention thực sự
        attn = self.dropout(attn_qk)

        out = torch.matmul(attn, v)               # dùng trọng số để tổng hợp value
        out = rearrange(out, 'b h n d -> b n (h d)')   # ghép các đầu lại
        # Trả về 3 thứ: output đã tổng hợp, VÀ 2 ma trận điểm thô để distill
        return self.to_out(out), dots_qk, dots_vv

class GLProj(nn.Module):
    """GL = Group-wise Linear projection. "Phiên dịch" đặc trưng CNN (src_dim chiều)
    sang không gian đặc trưng của ViT (tgt_dim chiều, vd 768). Điểm đặc biệt: KHÔNG
    dùng chung 1 lớp Linear cho mọi vị trí, mà chia lưới thành nhiều NHÓM (theo bảng
    idx ở PHẦN A), mỗi nhóm có 1 lớp Linear riêng -> mềm dẻo hơn, khớp tốt hơn với ViT.
    """
    def __init__(self, src_dim=1024, tgt_dim=768, num_patch=196):
        super().__init__()
        self.tgt_dim = tgt_dim
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        # Số lớp Linear = số nhóm, tùy độ phân giải: 196 patch -> 16 nhóm; 49 -> 4; còn lại -> 1
        if num_patch == 196:
            num_fc = 16
        elif num_patch == 49:
            num_fc = 4
        else:
            num_fc = 1
        for i in range(num_fc):
            layers[f"fc_layer_{i}"] = nn.Linear(src_dim, tgt_dim)   # 1 lớp Linear cho mỗi nhóm
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        # out: tensor kết quả rỗng, shape (batch, số_ô, tgt_dim). LƯU Ý: hardcode .to('cuda')
        # -> class này BẮT BUỘC chạy trên GPU (chạy CPU sẽ lỗi khác device).
        out = torch.zeros((x.shape[0], x.shape[1], self.tgt_dim)).to('cuda')
        num_fc = len(self.layers)
        # Chọn bảng nhóm tương ứng số lớp Linear
        if num_fc == 16:
            idx = idx_196
        elif num_fc == 4:
            idx = idx_49
        else:
            idx = None
        if idx is None:
            return self.layers[0](x)   # trường hợp 1 nhóm: dùng chung 1 Linear cho tất cả
        else:
            # Với mỗi nhóm i: lấy đúng các ô thuộc nhóm đó, đưa qua Linear riêng, ghi vào out
            for i in range(num_fc):
                out[:, idx[i], :] = self.layers[i](x[:, idx[i], :])
            return out


# =============================================================================
# PHẦN C — XƯƠNG SỐNG ResNet CHUẨN (giống hệt torchvision, không đổi)
# =============================================================================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """Conv 3x3 có padding — hàm rút gọn để đỡ lặp code. Đây là conv "trích đặc trưng" chính."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Conv 1x1 — chỉ đổi số kênh (không nhìn vùng lân cận), dùng để "nắn" số chiều
    trong bottleneck và ở nhánh downsample (shortcut)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Khối cơ bản của ResNet-18/34: 2 lớp conv 3x3 + kết nối tắt (residual/shortcut).
    "Residual" = học phần DƯ (out) rồi cộng lại đầu vào (identity): y = F(x) + x.
    Nhờ vậy mạng rất sâu vẫn train được (gradient không bị tiêu biến)."""
    expansion: int = 1   # hệ số nở kênh ở đầu ra = 1 (BasicBlock không nở kênh)

    def __init__(
        self,
        inplanes: int,     # số kênh đầu vào
        planes: int,       # số kênh "gốc" của khối
        stride: int = 1,   # bước nhảy (stride=2 -> giảm nửa kích thước không gian)
        downsample: Optional[nn.Module] = None,   # nhánh nắn shortcut khi shape lệch
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   # mặc định chuẩn hóa theo batch
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # conv1 (và downsample) là nơi giảm kích thước khi stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)   # conv 3x3 thứ nhất
        self.bn1 = norm_layer(planes)                    # chuẩn hóa
        self.relu = nn.ReLU(inplace=True)                # kích hoạt (inplace=True để tiết kiệm RAM)
        self.conv2 = conv3x3(planes, planes)             # conv 3x3 thứ hai
        self.bn2 = norm_layer(planes)
        self.downsample = downsample   # nếu cần: nắn identity cho khớp shape của out
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x   # giữ lại đầu vào cho kết nối tắt

        out = self.conv1(x)     # nhánh chính: conv -> bn -> relu
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # conv -> bn (chưa relu vội)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)   # nắn identity nếu shape đầu ra khác đầu vào

        out += identity   # <-- CỘNG residual: đây là điểm mấu chốt của ResNet
        out = self.relu(out)   # relu sau khi cộng

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # Bottleneck của ResNet-50/101/152: 3 lớp conv (1x1 -> 3x3 -> 1x1) theo kiểu "thắt cổ chai":
    # 1x1 giảm kênh -> 3x3 xử lý -> 1x1 nở kênh gấp 4. Tiết kiệm tính toán ở mạng sâu.
    expansion: int = 4   # đầu ra nở gấp 4 lần số kênh gốc

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups   # số kênh phần "thắt" (đổi khi dùng ResNeXt/WideResNet)
        # conv2 (và downsample) là nơi giảm kích thước khi stride != 1 (ResNet V1.5)
        self.conv1 = conv1x1(inplanes, width)                # 1x1: giảm kênh
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)   # 3x3: xử lý không gian
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) # 1x1: nở kênh gấp 4
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x   # giữ đầu vào cho shortcut

        out = self.conv1(x)     # 1x1 giảm kênh
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # 3x3 xử lý
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)   # 1x1 nở kênh
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity   # cộng residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet CHUẨN (bản gốc torchvision) — dùng cho student thường / baseline.
    Bản CAKD nằm ở class ResNet_CAKD phía dưới. Class này chỉ trả về logits phân loại."""
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        # "Cổ vào" (stem): conv 7x7 stride 2 -> 224x224 còn 112x112, đổi 3 kênh RGB thành 64 kênh
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 112x112 -> 56x56
        # 4 tầng chính, mỗi tầng gồm nhiều block; layers[i] = số block ở tầng i (vd [2,2,2,2] cho ResNet-18)
        self.layer1 = self._make_layer(block, 64, layers[0])                               # 56x56, 64 kênh
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])  # 28x28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])  # 14x14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])  # 7x7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))          # gộp trung bình toàn không gian -> vector 1 chiều
        self.fc = nn.Linear(512 * block.expansion, num_classes)   # lớp phân loại cuối -> logits cho mỗi lớp

        # Khởi tạo trọng số: conv theo kiểu Kaiming (hợp với ReLU), BN weight=1/bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Mẹo: khởi tạo BN cuối mỗi nhánh residual = 0 -> ban đầu block hoạt động như "đi thẳng"
        # (identity), giúp train ổn định, tăng ~0.2-0.3% (theo bài báo 1706.02677).
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],   # loại block (Basic hay Bottleneck)
        planes: int,     # số kênh gốc của tầng
        blocks: int,     # số block trong tầng
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Tạo 1 TẦNG = xếp chồng 'blocks' khối. Block ĐẦU có thể giảm kích thước
        (stride) và đổi kênh -> cần downsample cho shortcut; các block sau giữ nguyên."""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # Nếu kích thước/kênh đầu-cuối lệch nhau -> tạo nhánh downsample (conv 1x1 + norm) để shortcut khớp
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Block đầu tiên: nhận stride + downsample
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion   # cập nhật số kênh cho block kế tiếp
        for _ in range(1, blocks):
            # Các block còn lại: stride=1, không downsample (giữ nguyên shape)
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)   # gộp các block thành 1 tầng tuần tự

    def _forward_impl(self, x: Tensor) -> Tensor:
        # Luồng xuôi chuẩn: ảnh -> stem -> 4 tầng -> gộp -> phân loại
        x = self.conv1(x)      # 224->112, 3 kênh -> 64 kênh
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 112 -> 56

        x = self.layer1(x)     # 56x56
        x = self.layer2(x)     # 28x28
        x = self.layer3(x)     # 14x14
        x = self.layer4(x)     # 7x7

        x = self.avgpool(x)         # 7x7 -> 1x1
        x = torch.flatten(x, 1)     # duỗi thành vector (batch, kênh)
        x = self.fc(x)              # -> logits phân loại

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    """Hàm "xưởng" dựng ResNet chuẩn: tạo model + (nếu có) nạp trọng số pretrain."""
    if weights is not None:
        # Nếu dùng trọng số pretrain -> đặt num_classes khớp số lớp của bộ trọng số
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))   # nạp trọng số (khớp NGHIÊM NGẶT)

    return model


# =============================================================================
# PHẦN D — ResNet_CAKD: STUDENT ĐÃ ĐỘ THÊM NHÁNH DISTILL (QUAN TRỌNG NHẤT FILE)
# -----------------------------------------------------------------------------
# So với ResNet chuẩn, khác biệt nằm ở:
#   (1) __init__ thêm 3 lớp "phiên dịch": pca_proj (Attention), gl_proj (GLProj),
#       cls_proj (Linear) — đặt giữa layer3 và layer4.
#   (2) forward TRẢ VỀ 4 THỨ thay vì 1: (logits, [attn_qk, attn_vv], vit_feat, cls_token)
#       -> để file train dùng làm tín hiệu so khớp với teacher ViT.
# =============================================================================
class ResNet_CAKD(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        tgt_dim: int = 768,
        num_patch: int = 196,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # >>> 3 LỚP "PHIÊN DỊCH" THÊM VÀO cho CAKD (đặt ngay sau layer3) <<<
        # pca_proj: khối Attention chạy trên feature map layer3 -> sinh attention map giống ViT.
        #           self.inplanes lúc này = số kênh đầu ra của layer3; chia 16 head.
        self.pca_proj = Attention(dim=self.inplanes, heads=16, dim_head=int(self.inplanes/16))
        # gl_proj: chiếu feature CNN (self.inplanes chiều) sang chiều đặc trưng ViT (tgt_dim, vd 768)
        self.gl_proj = GLProj(src_dim=self.inplanes, tgt_dim=tgt_dim, num_patch=num_patch)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)   # đầu phân loại (logits)
        # cls_proj: chiếu vector đặc trưng cuối của CNN sang chiều ViT -> so khớp với "class token" của teacher
        self.cls_proj = nn.Linear(512 * block.expansion, tgt_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # ---- Phần xương sống CNN giống ResNet thường tới hết layer3 ----
        x = self.conv1(x)     # stem: 224->112, RGB -> 64 kênh
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # 112 -> 56

        x = self.layer1(x)    # 56x56
        x = self.layer2(x)    # 28x28
        x_3 = self.layer3(x)  # 14x14  <-- GIỮ LẠI đầu ra layer3 để làm tín hiệu distill

        # ---- NHÁNH DISTILL: biến feature map CNN thành "chuỗi token kiểu ViT" ----
        # x_3 có shape (batch, kênh C, 14, 14). Duỗi 14x14 = 196 -> (batch, C, 196)
        tmp = torch.reshape(x_3, (x_3.shape[0], x_3.shape[1], -1))
        # Đổi trục thành (batch, 196, C): mỗi ô lưới = 1 "token" C chiều -> đúng định dạng ViT nhận
        tmp = tmp.permute((0,2,1))
        # Đưa qua Attention: lấy 2 ma trận attention thô (bỏ output _), đây là thứ để so với teacher
        _, attn_qk, attn_vv = self.pca_proj(tmp)
        num_heads = attn_qk.shape[1]                 # số đầu attention
        attn_qk = attn_qk.sum(dim=1) / num_heads     # trung bình cộng các đầu -> 1 map attention (batch,196,196)
        attn_vv = attn_vv.sum(dim=1) / num_heads     # tương tự cho attn_vv
        vit_feat = self.gl_proj(tmp)                 # chiếu feature CNN sang chiều ViT (batch,196,768)

        # ---- Tiếp tục xương sống CNN cho nhánh phân loại ----
        x = self.layer4(x_3)   # 14x14 -> 7x7 (dùng LẠI x_3, không phải x của nhánh distill)

        x = self.avgpool(x)              # 7x7 -> 1x1
        cnn_token = torch.flatten(x, 1)  # vector đặc trưng cuối (batch, 512*expansion)
        x = self.fc(cnn_token)           # -> logits phân loại

        # TRẢ VỀ 4 THỨ (đây là điểm khác cốt lõi so với ResNet thường):
        #   x                    -> logits phân loại (dùng cho cls_loss + so logits với teacher)
        #   [attn_qk, attn_vv]   -> 2 attention map (dùng cho pca_loss + đưa vào discriminator GAN)
        #   vit_feat             -> feature theo patch đã chiếu sang chiều ViT (so với feature teacher)
        #   self.cls_proj(...)   -> class-token đã chiếu (so với class token của teacher)
        return x, [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_cakd(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet_CAKD:
    """Hàm "xưởng" dựng ResNet_CAKD. Giống _resnet nhưng nạp trọng số với strict=False."""
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_CAKD(block, layers, **kwargs)

    if weights is not None:
        # strict=False: CHO PHÉP lệch key. Bắt buộc vì model có thêm pca_proj/gl_proj/cls_proj
        # mà trọng số pretrain gốc KHÔNG có -> các lớp mới này giữ khởi tạo ngẫu nhiên.
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)

    return model


# =============================================================================
# PHẦN E — METADATA TRỌNG SỐ PRETRAIN & HÀM FACTORY (boilerplate của torchvision)
# -----------------------------------------------------------------------------
# Toàn bộ khối bên dưới KHÔNG chứa logic mạng — chỉ là "danh bạ" khai báo:
#   - Các class *_Weights: mỗi cái liệt kê URL tải file .pth trọng số pretrain trên
#     ImageNet, cách tiền xử lý ảnh (crop_size), và độ chính xác đã công bố (acc@1/@5).
#   - Các hàm resnet18/34/50/... : hàm tiện lợi để tạo model đúng cấu hình.
#     Trong đó [2,2,2,2]=ResNet18, [3,4,6,3]=ResNet34/50, [3,4,23,3]=ResNet101...
#   - Hai hàm QUAN TRỌNG cho CAKD: resnet18_cakd và resnet50_cakd (tạo student CAKD).
# Vì đây là mẫu lặp lại y hệt nhau nên chỉ comment ở các điểm cần chú ý, không dòng-từng-dòng.
# =============================================================================

_COMMON_META = {   # metadata dùng chung cho mọi bộ trọng số (kích thước tối thiểu, danh sách lớp)
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet34_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 21797672,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNet101_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 44549160,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNet152_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 60192808,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt50_32X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_32X8D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88791336,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_64X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 83455272,
            "recipe": "https://github.com/pytorch/vision/pull/5935",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                }
            },
            "_docs": """
                These weights were trained from scratch by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class Wide_ResNet101_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 126886696,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18_cakd(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_CAKD:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    # Tạo STUDENT CAKD dựa trên ResNet-18 (4 tầng, mỗi tầng 2 BasicBlock).
    # Dùng lại trọng số pretrain của resnet18 thường (phần trùng khớp), phần mới khởi tạo ngẫu nhiên.
    return _resnet_cakd(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50_cakd(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet_CAKD:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    # Tạo STUDENT CAKD dựa trên ResNet-50 (dùng Bottleneck, cấu hình tầng [3,4,6,3]).
    return _resnet_cakd(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet101_Weights.IMAGENET1K_V1))
def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet152_Weights.IMAGENET1K_V1))
def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.IMAGENET1K_V1))
def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnext101_64x4d(
    *, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """
    weights = ResNeXt101_64X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.IMAGENET1K_V1))
def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


# The dictionary below is internal implementation detail and will be removed in v0.15
from ._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1.url,
        "resnet34": ResNet34_Weights.IMAGENET1K_V1.url,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1.url,
        "resnet101": ResNet101_Weights.IMAGENET1K_V1.url,
        "resnet152": ResNet152_Weights.IMAGENET1K_V1.url,
        "resnext50_32x4d": ResNeXt50_32X4D_Weights.IMAGENET1K_V1.url,
        "resnext101_32x8d": ResNeXt101_32X8D_Weights.IMAGENET1K_V1.url,
        "wide_resnet50_2": Wide_ResNet50_2_Weights.IMAGENET1K_V1.url,
        "wide_resnet101_2": Wide_ResNet101_2_Weights.IMAGENET1K_V1.url,
    }
)
