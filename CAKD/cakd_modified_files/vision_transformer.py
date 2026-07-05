# =============================================================================
# vision_transformer.py — TEACHER MODEL của hệ thống CAKD (ViT đã được "độ")
# -----------------------------------------------------------------------------
# Đây là bản chép từ torchvision/models/vision_transformer.py rồi SỬA cho CAKD.
# ViT (Vision Transformer) là model TO, GIỎI, đã pretrain sẵn -> đóng vai THẦY.
# Khi train CAKD, teacher này ở chế độ .eval() (đóng băng, KHÔNG học), chỉ chạy
# xuôi để "phát tín hiệu" cho student (ResNet_CAKD) bắt chước.
#
# ViT hoạt động khác CNN: nó CẮT ảnh thành các ô vuông (patch), coi mỗi patch
# như 1 "từ" (token), rồi dùng cơ chế self-attention để các token "nhìn nhau".
#
# ĐIỂM SỬA CHO CAKD (so với ViT gốc torchvision) — chỉ 3 chỗ:
#   (1) EncoderBlock.forward : thêm cờ need_weights -> cho phép TRẢ RA ma trận attention.
#   (2) Encoder.forward      : MOI attention của 2 block cuối + trả về danh sách attention.
#   (3) VisionTransformer.forward: trả về 4 THỨ (logits, attention, cls_token, feats)
#                                  thay vì chỉ logits -> làm "đáp án mẫu" cho student.
# Phần còn lại (Weights, factory vit_b_16...) là boilerplate chuẩn torchvision.
# =============================================================================

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Dict

import torch
import torch.nn as nn

# ----- Import nội bộ torchvision (đường dẫn ".." -> file phải nằm trong cây torchvision/models) -----
from ..ops.misc import Conv2dNormActivation, MLP   # khối conv+norm+act và khối MLP dựng sẵn
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


# ConvStemConfig: cấu hình cho biến thể "conv stem" (dùng conv thay vì cắt patch thẳng).
# Chỉ dùng khi truyền conv_stem_configs; CAKD dùng bản mặc định nên phần này ít khi động tới.
class ConvStemConfig(NamedTuple):
    out_channels: int                 # số kênh đầu ra của lớp conv
    kernel_size: int                  # kích thước kernel
    stride: int                       # bước nhảy
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d      # lớp chuẩn hóa
    activation_layer: Callable[..., nn.Module] = nn.ReLU       # hàm kích hoạt


class MLPBlock(MLP):
    """Khối MLP trong mỗi lớp Transformer: Linear -> GELU -> Dropout -> Linear -> Dropout.
    Kế thừa lớp MLP dựng sẵn của torchvision; đây là phần "xử lý riêng từng token" sau attention."""

    _version = 2   # đánh phiên bản để xử lý tương thích khi nạp trọng số cũ (xem _load_from_state_dict)

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        # MLP nở từ in_dim -> mlp_dim rồi thu về in_dim, kích hoạt GELU
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        # Khởi tạo trọng số cho các lớp Linear: weight theo Xavier, bias gần 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Hàm tương thích ngược: đổi tên key trọng số kiểu CŨ (linear_1/linear_2) sang kiểu MỚI
        # để trọng số pretrain cũ vẫn nạp được. Không liên quan logic mạng, có thể bỏ qua khi đọc.
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """MỘT lớp Transformer encoder = [Self-Attention] + [MLP], mỗi phần có LayerNorm và
    kết nối tắt (residual). ViT-B/16 xếp chồng 12 lớp như thế này.
    >>> ĐÃ SỬA CHO CAKD: forward thêm tham số need_weights để (khi cần) TRẢ RA ma trận attention.
    """

    def __init__(
        self,
        num_heads: int,     # số đầu attention
        hidden_dim: int,    # chiều đặc trưng của mỗi token (ViT-B = 768)
        mlp_dim: int,       # chiều ẩn của khối MLP (ViT-B = 3072)
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # --- Khối Attention ---
        self.ln_1 = norm_layer(hidden_dim)   # LayerNorm trước attention (pre-norm)
        # MultiheadAttention: cơ chế để mỗi token "chú ý" tới các token khác. batch_first=True
        # -> tensor có dạng (batch, seq, dim).
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # --- Khối MLP ---
        self.ln_2 = norm_layer(hidden_dim)   # LayerNorm trước MLP
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, need_weights: bool=False):
        # input phải là 3 chiều: (batch, số_token, hidden_dim)
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)   # chuẩn hóa
        # Self-attention: query=key=value=x (mỗi token nhìn tất cả token). need_weights=True thì
        # attn_weights = ma trận "ai chú ý ai" (đây chính là thứ CAKD cần để distill).
        x, attn_weights = self.self_attention(query=x, key=x, value=x, need_weights=need_weights)
        x = self.dropout(x)
        x = x + input          # residual quanh khối attention

        y = self.ln_2(x)       # chuẩn hóa
        y = self.mlp(y)        # MLP xử lý từng token
        if not need_weights:
            return x + y        # bình thường: chỉ trả về output (x+y = residual quanh khối MLP)
        return x + y, attn_weights   # khi cần: trả về CẢ output VÀ ma trận attention


class Encoder(nn.Module):
    """Bộ Encoder = xếp chồng num_layers khối EncoderBlock + thêm "vị trí" (pos_embedding).
    >>> ĐÃ SỬA CHO CAKD: forward MOI ra ma trận attention của 2 lớp CUỐI để làm tín hiệu distill.
    """

    def __init__(
        self,
        seq_length: int,    # số token (số patch + 1 class token)
        num_layers: int,    # số lớp Transformer (ViT-B = 12)
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # pos_embedding: "nhãn vị trí" học được, cộng vào token để model biết token nào ở đâu
        # (attention vốn không có khái niệm thứ tự -> phải thêm vị trí). Khởi tạo kiểu BERT.
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(   # tạo num_layers khối giống nhau
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)   # LayerNorm cuối cùng

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding   # cộng thông tin vị trí vào các token
        num_layers = len(self.layers)
        x = self.dropout(input)
        # Chạy lần lượt qua từng lớp. Với 2 lớp CUỐI thì bật need_weights=True để lấy attention:
        for i in range(num_layers):
            if i < num_layers - 2:
                x = self.layers[i](x)                          # các lớp giữa: chỉ lấy output
            elif i == num_layers-2:
                x, attn_weights_2 = self.layers[i](x, True)    # lớp áp chót: lấy cả attention (attn_weights_2)
            else:
                x, attn_weights_1 = self.layers[i](x, True)    # lớp cuối: lấy cả attention (attn_weights_1)
        # Trả về: (đặc trưng đã chuẩn hóa, DANH SÁCH 4 ma trận attention từ 2 lớp cuối).
        # Danh sách này chính là tea_attn_weights mà student so khớp (xem dist_train_cakd.py).
        # Lưu ý: cách đánh index [0]/[1] tách ra các tensor attention cụ thể trong mỗi lớp.
        return self.ln(x), [attn_weights_2[0], attn_weights_2[1], attn_weights_1[0], attn_weights_1[1]]
        #return self.ln(self.layers(self.dropout(input)))   # <- dòng gốc của ViT thường (đã thay bằng ở trên)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) theo bài báo https://arxiv.org/abs/2010.11929.
    Luồng: cắt ảnh thành patch -> nhúng thành token -> thêm class token & vị trí -> qua Encoder
    (nhiều lớp attention) -> lấy class token đưa vào lớp phân loại.
    >>> ĐÃ SỬA CHO CAKD: forward trả về 4 thứ (logits, attention, cls_token, feats)."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            # conv_proj: cách "cắt patch" chuẩn của ViT. Conv với kernel=stride=patch_size nghĩa là
            # mỗi ô patch (vd 16x16) được nén thành 1 vector hidden_dim chiều -> chính là "token nhúng".
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        # Số patch = (224/16)^2 = 14*14 = 196 (với ViT-B/16)
        seq_length = (image_size // patch_size) ** 2

        # class_token: 1 token "đặc biệt" học được, gắn thêm vào đầu chuỗi. Sau khi qua Encoder,
        # token này tổng hợp thông tin toàn ảnh -> dùng để phân loại. seq_length +1 = 197.
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(   # bộ encoder nhiều lớp attention
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        # heads: "đầu" phân loại. Thường chỉ 1 lớp Linear: hidden_dim -> num_classes (số lớp).
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            # Biến thể có thêm lớp trung gian (pre_logits) + Tanh trước khi phân loại
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # ----- Khởi tạo trọng số ban đầu cho các lớp (không ảnh hưởng logic, chỉ giúp train tốt) -----
        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Biến ảnh (batch, 3, 224, 224) thành CHUỖI TOKEN (batch, 196, hidden_dim)."""
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p   # số patch theo chiều cao = 14
        n_w = w // p   # số patch theo chiều rộng = 14

        # Cắt patch + nhúng: (n, 3, 224, 224) -> (n, hidden_dim, 14, 14)
        x = self.conv_proj(x)
        # Duỗi lưới 14x14 thành 196: (n, hidden_dim, 14, 14) -> (n, hidden_dim, 196)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # Đổi trục về (n, 196, hidden_dim): mỗi hàng = 1 token. Đây là định dạng (N, S, E)
        # mà lớp attention mong đợi (N=batch, S=số token, E=chiều nhúng).
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # 1) Ảnh -> chuỗi 196 token
        x = self._process_input(x)
        n = x.shape[0]

        # 2) Gắn class token vào ĐẦU chuỗi -> thành 197 token
        batch_class_token = self.class_token.expand(n, -1, -1)   # nhân bản class token cho cả batch
        x = torch.cat([batch_class_token, x], dim=1)

        # 3) Qua Encoder (nhiều lớp attention). attn_weights = danh sách 4 attention của 2 lớp cuối.
        x, attn_weights = self.encoder(x)

        # 4) Tách kết quả:
        cls_token = x[:, 0]    # token vị trí 0 = class token -> đại diện toàn ảnh (dùng phân loại + distill)
        feats = x[:, 1:]       # 196 token còn lại = đặc trưng theo từng patch (dùng distill với vit_feat của student)

        x = self.heads(cls_token)   # đưa class token qua đầu phân loại -> logits

        # TRẢ VỀ 4 THỨ (đây là điểm sửa cốt lõi so với ViT thường) — "đáp án mẫu" cho student:
        #   x           -> logits (student khớp qua gl_loss + so logits)
        #   attn_weights-> 4 attention map của 2 lớp cuối (student khớp qua pca_loss + đưa vào GAN)
        #   cls_token   -> class token (student khớp bằng cls_proj)
        #   feats       -> feature theo patch (student khớp bằng vit_feat/gl_proj)
        return x, attn_weights, cls_token, feats


# =============================================================================
# PHẦN BOILERPLATE (chuẩn torchvision) — hàm factory + metadata trọng số pretrain
# -----------------------------------------------------------------------------
# Từ đây tới cuối file gần như KHÔNG có logic mạng, chỉ khai báo:
#   - _vision_transformer: hàm "xưởng" dựng model + nạp trọng số.
#   - Các class ViT_*_Weights: URL tải trọng số pretrain + độ chính xác công bố.
#   - Các hàm vit_b_16 / vit_l_16 / ...: tạo ViT đúng cấu hình. CAKD dùng teacher là
#     một trong số này (thường vit_b_16: patch=16, 12 lớp, 12 head, hidden=768).
#   - interpolate_embeddings: nội suy pos_embedding khi đổi độ phân giải ảnh.
# =============================================================================

def _vision_transformer(
    patch_size: int,     # kích thước 1 patch (16 -> ảnh 224 thành 14x14 patch)
    num_layers: int,     # số lớp Transformer
    num_heads: int,      # số đầu attention
    hidden_dim: int,     # chiều đặc trưng token
    mlp_dim: int,        # chiều ẩn MLP
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer:
    """Hàm "xưởng" dựng một VisionTransformer theo cấu hình + (nếu có) nạp trọng số pretrain."""
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}


class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16-c867db91.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 86567656,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
        transforms=partial(
            ImageClassification,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 86859496,
            "min_size": (384, 384),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 86567656,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 88224232,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=242),
        meta={
            **_COMMON_META,
            "num_params": 304326632,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
        transforms=partial(
            ImageClassification,
            crop_size=512,
            resize_size=512,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 305174504,
            "min_size": (512, 512),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 304326632,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vit_l_32-c7638314.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 306535400,
            "min_size": (224, 224),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                }
            },
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
        transforms=partial(
            ImageClassification,
            crop_size=518,
            resize_size=518,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "num_params": 633470440,
            "min_size": (518, 518),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                }
            },
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        url="https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
        transforms=partial(
            ImageClassification,
            crop_size=224,
            resize_size=224,
            interpolation=InterpolationMode.BICUBIC,
        ),
        meta={
            **_COMMON_SWAG_META,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "num_params": 632045800,
            "min_size": (224, 224),
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                }
            },
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_SWAG_E2E_V1


@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    # ViT-Base/16: patch 16x16, 12 lớp Transformer, 12 đầu attention, token 768 chiều.
    # Đây là cấu hình teacher điển hình của CAKD (hidden_dim=768 khớp tgt_dim của student).
    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", ViT_B_32_Weights.IMAGENET1K_V1))
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    weights = ViT_B_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", ViT_L_16_Weights.IMAGENET1K_V1))
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    weights = ViT_L_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", ViT_L_32_Weights.IMAGENET1K_V1))
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    weights = ViT_L_32_Weights.verify(weights)

    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = None, progress: bool = True, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    weights = ViT_H_14_Weights.verify(weights)

    return _vision_transformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """Nội suy "nhãn vị trí" (pos_embedding) khi nạp trọng số pretrain vào model có ĐỘ PHÂN GIẢI
    ẢNH KHÁC (số patch khác -> số vị trí khác). Hàm này co giãn bảng vị trí cũ cho khớp kích thước
    mới bằng nội suy 2D (giữ nguyên vị trí class token). Chỉ là tiện ích nạp checkpoint, không phải
    logic mạng lúc train CAKD.
    ---
    This function helps interpolating positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state


# The dictionary below is internal implementation detail and will be removed in v0.15
from ._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1.url,
        "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1.url,
        "vit_l_16": ViT_L_16_Weights.IMAGENET1K_V1.url,
        "vit_l_32": ViT_L_32_Weights.IMAGENET1K_V1.url,
    }
)
