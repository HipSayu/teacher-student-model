# Chặng 2 — `CAKD/cakd_modified_files/resnet.py` (HỌC SINH)

> File dài ~1260 dòng nhưng **90% là ResNet gốc của torchvision**. CAKD chỉ thêm 4 mảnh:
> `Attention` · `GLProj` · `ResNet_CAKD` · `resnet50_cakd`. Chỉ đọc kỹ 4 mảnh này.

---

## Bối cảnh: học sinh phải trả về 4 thứ

```python
return x, [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)   # dòng 550
#      logits   ↑PCA↑          ↑GL-feat↑    ↑GL-token↑
```
Ba mảnh dưới đây chính là nơi sinh ra 3 thứ "lạ" (ngoài logits).

---

## 🟢 Bảng `idx_*` — dòng 44–66

```python
idx_224_16_0 = [0,1,2,3,14,15,16,17, ...]   # danh sách chỉ số patch thuộc NHÓM 0
...
idx_196 = [idx_224_16_0, ..., idx_224_16_15]  # gộp 196 patch thành 16 NHÓM không gian (dùng cho ảnh 224 → 14×14=196 patch)
idx_49  = [idx_224_32_0, ..., idx_224_32_3]   # phương án 49 patch → 4 nhóm
```
> Đây là "bản đồ chia nhóm" cho **Group-wise Linear**. Mỗi nhóm gồm các patch gần nhau về không gian,
> sẽ được xử lý bởi **một** lớp Linear riêng. Không cần thuộc lòng con số — chỉ cần hiểu: *196 patch → 16 nhóm*.

---

## 🟢 `Attention` — dòng 90–121  (lõi của PCA)

Khối self-attention gắn thêm vào ResNet để sinh ra "bản đồ chú ý" kiểu Transformer.

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads                 # tổng chiều sau khi chia đầu
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5                # hệ số 1/√d để ổn định softmax
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)   # 1 Linear sinh ra CẢ Q, K, V (gộp)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):                            # x: (B, N=196, C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)        # tách khối gộp thành (Q, K, V)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # chia 16 đầu
        dots_qk = torch.matmul(q, k.transpose(-1,-2)) * self.scale   # Q·Kᵀ/√d  = chú ý cổ điển  (B,h,N,N)
        dots_vv = torch.matmul(v, v.transpose(-1,-2)) * self.scale   # V·Vᵀ/√d  = "Gram" của Value (B,h,N,N)
        attn_qk = self.attend(dots_qk)               # softmax để lấy output (nhánh này KHÔNG dùng cho KD)
        attn = self.dropout(attn_qk)
        out = torch.matmul(attn, v)                  # attention output thông thường
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), dots_qk, dots_vv    # ← TRẢ VỀ dots_qk, dots_vv (TRƯỚC softmax) cho KD
```
> **3 điều phải khắc cốt:**
> 1. Trả về **2 ma trận**: `dots_qk` (Q·K) và `dots_vv` (V·V) — CAKD khớp cả hai với giáo viên.
> 2. Trả về **điểm số THÔ (trước softmax)** — vì giáo viên cũng trả điểm thô, hai bên khớp cùng "thang".
> 3. `out` (output attention thường) hầu như bị bỏ (`_`) — cái ta cần chỉ là 2 ma trận điểm.

---

## 🟢 `GLProj` — dòng 123–152  (lõi của GL = Group-wise Linear)

"Phiên dịch viên" chuyển đặc trưng ResNet (chiều 1024) sang không gian đặc trưng ViT (chiều 768), **theo từng nhóm patch**.

```python
class GLProj(nn.Module):
    def __init__(self, src_dim=1024, tgt_dim=768, num_patch=196):
        super().__init__()
        self.tgt_dim = tgt_dim
        layers = OrderedDict()
        if num_patch == 196:  num_fc = 16            # 196 patch → 16 nhóm → 16 Linear
        elif num_patch == 49: num_fc = 4
        else:                 num_fc = 1
        for i in range(num_fc):
            layers[f"fc_layer_{i}"] = nn.Linear(src_dim, tgt_dim)   # MỖI NHÓM một Linear KHÁC NHAU
        self.layers = nn.Sequential(layers)

    def forward(self, x):                            # x: (B, 196, 1024)
        out = torch.zeros((x.shape[0], x.shape[1], self.tgt_dim)).to('cuda')  # khung chứa kết quả (B,196,768)
        num_fc = len(self.layers)
        idx = idx_196 if num_fc == 16 else (idx_49 if num_fc == 4 else None)
        if idx is None:
            return self.layers[0](x)                 # trường hợp 1 Linear chung (num_patch khác)
        for i in range(num_fc):
            out[:, idx[i], :] = self.layers[i](x[:, idx[i], :])   # nhóm i ← Linear i (chỉ chiếu các patch của nhóm đó)
        return out                                   # (B, 196, 768) = "vit_feat"
```
> **Vì sao "group-wise" thay vì 1 Linear chung?** Các vùng khác nhau của ảnh có phân bố đặc trưng khác nhau;
> cho mỗi nhóm một phép chiếu riêng thì "dịch" sang không gian ViT sát hơn, mà vẫn nhẹ.
> ⚠️ `.to('cuda')` bị hard-code → chỉ chạy được trên GPU (đọc để biết, không phải bug của mày).

---

## 🟢 `ResNet_CAKD` — dòng 420–553

### `__init__` (420–482) — điểm khác ResNet gốc
```python
self.layer1 = self._make_layer(block, 64, layers[0])                       # giống ResNet
self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ...)
self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ...)
self.pca_proj = Attention(dim=self.inplanes, heads=16, dim_head=inplanes/16)  # ← THÊM: khối attention (PCA)
self.gl_proj  = GLProj(src_dim=self.inplanes, tgt_dim=tgt_dim, num_patch=num_patch)  # ← THÊM: chiếu nhóm (GL)
self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ...)        # giống ResNet
self.avgpool = nn.AdaptiveAvgPool2d((1,1))
self.fc = nn.Linear(512*block.expansion, num_classes)                       # đầu phân loại
self.cls_proj = nn.Linear(512*block.expansion, tgt_dim)                     # ← THÊM: chiếu token CNN→768 (GL-token)
```
> So với ResNet gốc: thêm **3 thành phần** `pca_proj`, `gl_proj`, `cls_proj`. `pca_proj`/`gl_proj` cắm vào **giữa** (sau layer3),
> `cls_proj` cắm ở **cuối** (sau avgpool). Phần khởi tạo trọng số (Kaiming/BN) là code gốc → ⚪ lướt.

### `_forward_impl` (525–553) — luồng dữ liệu, ĐỌC KỸ
```python
def _forward_impl(self, x):
    x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)  # stem (gốc)
    x = self.layer1(x)
    x = self.layer2(x)
    x_3 = self.layer3(x)                             # ← đặc trưng tại layer3: (B, C=1024, 14, 14)

    # ── NHÁNH RẼ: từ x_3 sinh ra PCA + GL ──
    tmp = torch.reshape(x_3, (x_3.shape[0], x_3.shape[1], -1))  # (B, 1024, 196)  gộp H×W=196
    tmp = tmp.permute((0,2,1))                        # (B, 196, 1024)  đưa patch lên trục "n" (dạng token)
    _, attn_qk, attn_vv = self.pca_proj(tmp)          # chạy attention → 2 ma trận (B,16,196,196)
    num_heads = attn_qk.shape[1]
    attn_qk = attn_qk.sum(dim=1) / num_heads          # gộp 16 đầu → (B,196,196)
    attn_vv = attn_vv.sum(dim=1) / num_heads          # (B,196,196)
    vit_feat = self.gl_proj(tmp)                      # chiếu nhóm → (B,196,768)  = proj_feat

    # ── NHÁNH CHÍNH: tiếp tục ResNet để phân loại ──
    x = self.layer4(x_3)                              # dùng x_3 (KHÔNG dùng tmp) chạy tiếp
    x = self.avgpool(x)
    cnn_token = torch.flatten(x, 1)                   # (B, 2048) vector đặc trưng ảnh
    x = self.fc(cnn_token)                            # (B, n_cls) logits

    return x, [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)
    #      logits    PCA            GL-feat    GL-token (cnn_token chiếu sang 768)
```
> **Sơ đồ rẽ nhánh:**
> ```
>            x_3 (sau layer3)
>            ├──► reshape+permute → tmp ─┬─► pca_proj → attn_qk, attn_vv   (PCA)
>            │                           └─► gl_proj  → vit_feat            (GL-feat)
>            └──► layer4 → avgpool → cnn_token ─┬─► fc       → logits
>                                               └─► cls_proj → proj_token   (GL-token)
> ```
> Nhớ: nhánh PCA/GL **không** chảy ngược vào layer4 — nó là "chi nhánh phụ" chỉ để lấy tín hiệu khớp giáo viên.

---

## 🟢 `resnet50_cakd` — dòng 1022+  (hàm factory)

```python
def resnet50_cakd(*, weights=None, progress=True, **kwargs):
    ...
    return _resnet_cakd(Bottleneck, [3,4,6,3], weights, progress, **kwargs)  # ResNet50 = Bottleneck [3,4,6,3]
```
> Chỉ là "nhà máy" ráp `ResNet_CAKD` với cấu hình ResNet50. Trong train gọi `torchvision.models.resnet50_cakd(num_classes=...)`.
> ⚪ `_resnet_cakd` (556–571) và các `*_Weights` (580–700) là khung gốc → lướt.

---

## ⚪ Phần LƯỚT (không đọc từng dòng)
`conv3x3`/`conv1x1` · `BasicBlock` · `Bottleneck` · `ResNet` (gốc) · tất cả `*_Weights` · các factory `resnet18/34/101/152`, `resnext*`, `wide_resnet*`. Đây là torchvision nguyên bản.

---

## ✅ Kiểm tra hiểu
1. `model(image)` trả về mấy thứ, mỗi thứ shape gì, dùng cho loss nào?
2. Nhánh PCA/GL rẽ ra từ đâu (sau layer mấy)? Nó có ảnh hưởng nhánh phân loại không?
3. `GLProj` khác một `nn.Linear` chung ở điểm gì?
