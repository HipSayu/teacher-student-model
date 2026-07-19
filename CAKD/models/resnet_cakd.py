# =============================================================================
# resnet_cakd.py — STUDENT ResNet_CAKD (ban PORT sang torch 2.x, chay native).
# -----------------------------------------------------------------------------
# Khac ban trong cakd_modified_files/resnet.py o cho:
#   - KHONG import noi bo torchvision (bo _log_api_usage_once, WeightsEnum, ...).
#   - GLProj device-agnostic (bo hardcode .to('cuda')) -> chay CPU/GPU deu duoc.
#   - Nap pretrained ResNet-50 backbone qua torchvision.models.resnet50 (strict=False).
# Forward tra ve 4 thu: (logits, [attn_qk, attn_vv], vit_feat, cls_token) — giong ban goc.
# =============================================================================
from collections import OrderedDict
from typing import Type, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

# ----- Bang chi so anh xa luoi CNN (14x14=196) <-> nhom patch (dung boi GLProj) -----
idx_224_16_0 = [0,1,2,3,14,15,16,17,28,29,30,31,42,43,44,45]
idx_224_16_1 = [4,5,6,7,18,19,20,21,32,33,34,35,46,47,48,49]
idx_224_16_2 = [8,9,10,11,22,23,24,25,36,37,38,39,50,51,52,53]
idx_224_16_3 = [12,13,26,27,40,41,54,55]
idx_224_16_4 = [56,57,58,59,70,71,72,73,84,85,86,87,98,99,100,101]
idx_224_16_5 = [60,61,62,63,74,75,76,77,88,89,90,91,102,103,104,105]
idx_224_16_6 = [64,65,66,67,78,79,80,81,92,93,94,95,106,107,108,109]
idx_224_16_7 = [68,69,82,83,96,97,110,111]
idx_224_16_8 = [112,113,114,115,126,127,128,129,140,141,142,143,154,155,156,157]
idx_224_16_9 = [116,117,118,119,130,131,132,133,144,145,146,147,158,159,160,161]
idx_224_16_10 = [120,121,122,123,134,135,136,137,148,149,150,151,162,163,164,165]
idx_224_16_11 = [124,125,138,139,152,153,166,167]
idx_224_16_12 = [168,169,170,171,182,183,184,185]
idx_224_16_13 = [172,173,174,175,186,187,188,189]
idx_224_16_14 = [176,177,178,179,190,191,192,193]
idx_224_16_15 = [180,181,194,195]
idx_196 = [idx_224_16_0,idx_224_16_1,idx_224_16_2,idx_224_16_3,idx_224_16_4,idx_224_16_5,
           idx_224_16_6,idx_224_16_7,idx_224_16_8,idx_224_16_9,idx_224_16_10,idx_224_16_11,
           idx_224_16_12,idx_224_16_13,idx_224_16_14,idx_224_16_15]

idx_224_32_0 = [0,1,2,3,7,8,9,10,14,15,16,17,21,22,23,24]
idx_224_32_1 = [4,5,6,11,12,13,18,19,20,25,26,27]
idx_224_32_2 = [28,29,30,31,35,36,37,38,42,43,44,45]
idx_224_32_3 = [32,33,34,39,40,41,46,47,48]
idx_49 = [idx_224_32_0,idx_224_32_1,idx_224_32_2,idx_224_32_3]


class Attention(nn.Module):
    """Self-Attention gan len feature map CNN -> sinh attention map giong ViT.
    Tra ve output + 2 ma tran diem tho (dots_qk, dots_vv) de distill voi teacher."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots_qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots_vv = torch.matmul(v, v.transpose(-1, -2)) * self.scale
        attn_qk = self.attend(dots_qk)
        attn = self.dropout(attn_qk)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), dots_qk, dots_vv


class GLProj(nn.Module):
    """Group-wise Linear projection: chieu feature CNN (src_dim) sang chieu ViT (tgt_dim),
    moi nhom patch dung 1 Linear rieng. Device-agnostic (khac ban goc hardcode cuda)."""
    def __init__(self, src_dim=1024, tgt_dim=768, num_patch=196):
        super().__init__()
        self.tgt_dim = tgt_dim
        layers: "OrderedDict[str, nn.Module]" = OrderedDict()
        if num_patch == 196:
            num_fc = 16
        elif num_patch == 49:
            num_fc = 4
        else:
            num_fc = 1
        for i in range(num_fc):
            layers[f"fc_layer_{i}"] = nn.Linear(src_dim, tgt_dim)
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        out = torch.zeros((x.shape[0], x.shape[1], self.tgt_dim), device=x.device, dtype=x.dtype)
        num_fc = len(self.layers)
        if num_fc == 16:
            idx = idx_196
        elif num_fc == 4:
            idx = idx_49
        else:
            idx = None
        if idx is None:
            return self.layers[0](x)
        for i in range(num_fc):
            out[:, idx[i], :] = self.layers[i](x[:, idx[i], :])
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_CAKD(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, tgt_dim=768, num_patch=196):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # >>> 3 lop "phien dich" cho CAKD (dat sau layer3) <<<
        self.pca_proj = Attention(dim=self.inplanes, heads=16, dim_head=int(self.inplanes / 16))
        self.gl_proj = GLProj(src_dim=self.inplanes, tgt_dim=tgt_dim, num_patch=num_patch)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.cls_proj = nn.Linear(512 * block.expansion, tgt_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)          # 14x14 -> giu lai lam tin hieu distill
        # nhanh distill: feature map -> chuoi 196 token
        tmp = torch.reshape(x_3, (x_3.shape[0], x_3.shape[1], -1)).permute((0, 2, 1))
        _, attn_qk, attn_vv = self.pca_proj(tmp)
        num_heads = attn_qk.shape[1]
        attn_qk = attn_qk.sum(dim=1) / num_heads   # trung binh head -> (N,196,196)
        attn_vv = attn_vv.sum(dim=1) / num_heads
        vit_feat = self.gl_proj(tmp)               # (N,196,768)
        # nhanh phan loai
        x = self.layer4(x_3)
        x = self.avgpool(x)
        cnn_token = torch.flatten(x, 1)
        logits = self.fc(cnn_token)
        return logits, [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)


def resnet50_cakd(num_classes=1000, pretrained=False):
    """Tao ResNet_CAKD-50. pretrained=True -> nap backbone ImageNet (bo fc, strict=False)."""
    model = ResNet_CAKD(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        from torchvision.models import resnet50, ResNet50_Weights
        sd = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}  # fc khac so lop
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # pca_proj/gl_proj/cls_proj/fc nam trong 'missing' (giu khoi tao) -> hop le
    return model


def resnet18_cakd(num_classes=1000, pretrained=False):
    """Tao ResNet_CAKD-18 (BasicBlock, expansion=1) — student NHE hon resnet50.
    Khac resnet50 duy nhat o cho: kenh sau layer3 = 256 (thay vi 1024) va sau layer4 = 512
    (thay vi 2048). pca_proj/gl_proj/cls_proj tu dong bam theo self.inplanes/expansion nen
    KHONG can chinh gi them; luoi 14x14=196 patch giu nguyen -> khop teacher ViT nhu cu.
    pretrained=True -> nap backbone ResNet-18 ImageNet (bo fc, strict=False)."""
    model = ResNet_CAKD(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        from torchvision.models import resnet18, ResNet18_Weights
        sd = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}  # fc khac so lop
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # pca_proj/gl_proj/cls_proj/fc nam trong 'missing' (giu khoi tao) -> hop le
    return model
