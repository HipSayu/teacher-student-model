# `t_S` (class token của student) — từ ảnh gốc ra token qua những bước nào?

> Giải thích biến `proj_token` (cls_token) trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $t_S$ | `proj_token` (cls_token) | (B, 768) | [`resnet.py:666`](cakd_modified_files/resnet.py) `self.cls_proj(cnn_token)` | đầu ra `cls_proj` (Linear) |

`t_S` là **1 vector 768 chiều cho mỗi ảnh** (shape (B, 768)) — "bản tóm tắt toàn ảnh" của student,
cố ý làm cho **giống class token của teacher ViT** (`tea_token` cũng (B, 768)). Nó ra từ
**nhánh phân loại** (KHÁC với `f_S`/attention ra từ nhánh distill), qua 1 lớp `cls_proj`
(Linear 2048 → 768).

> 👉 Điểm khác biệt lớn nhất so với 3 đầu ra kia: `t_S` **KHÔNG** đi qua `tmp`/nhánh distill,
> mà rẽ ra từ **`cnn_token`** — đúng cái vector mà `fc` dùng để sinh logits `z_S`.

---

## Hành trình `image → t_S`

```
image                          (B,   3, 224, 224)   ← ảnh RGB đầu vào
  │
  ├─ stem (conv1+bn1+relu+maxpool) → (B,  64,  56,  56)   [633-636]
  ├─ layer1                        → (B, 256,  56,  56)   [638]
  ├─ layer2                        → (B, 512,  28,  28)   [639]
  ├─ layer3                        → (B,1024,  14,  14)   [640]  = x_3
  │
  │   ════════ NHÁNH PHÂN LOẠI (KHÔNG phải nhánh distill) ════════
  ├─ layer4(x_3)                   → (B,2048,   7,   7)   [655]  tầng CNN sâu nhất
  ├─ avgpool (AdaptiveAvgPool 1×1) → (B,2048,   1,   1)   [657]  gộp trung bình toàn ảnh 7×7 → 1 điểm
  ├─ flatten                       → (B,2048)             [658]  = cnn_token  ★ vector tóm tắt toàn ảnh
  │
  │        cnn_token rẽ 2 đường:
  ├─ self.fc(cnn_token)            → (B,1000)             [659]  = output = z_S   (logits)
  └─ self.cls_proj(cnn_token)      → (B, 768)             [666]  = proj_token = t_S  ★ CÁI TA QUAN TÂM
```

> `z_S` (logits) và `t_S` (class token) là **2 "đầu" mọc ra từ CÙNG 1 gốc `cnn_token`** —
> một đầu để phân loại (1000 lớp), một đầu để bắt chước class token teacher (768 chiều).

---

## `cnn_token` là gì? (đầu vào của `cls_proj`)

`cnn_token` ([`resnet.py:658`](cakd_modified_files/resnet.py)) là **vector đặc trưng toàn cục** của
cả ảnh, shape (B, 2048):

```
layer4 → (B, 2048, 7, 7)      ← 2048 kênh, mỗi kênh là lưới 7×7
  │  avgpool: lấy TRUNG BÌNH 49 ô của mỗi kênh
  ▼
(B, 2048, 1, 1)               ← mỗi kênh gộp còn 1 số
  │  flatten
  ▼
cnn_token (B, 2048)           ← 2048 số = "tóm tắt toàn ảnh"
```

`avgpool` (Adaptive Average Pooling) bóp lưới 7×7 của mỗi kênh xuống còn **1 số trung bình** →
gộp thông tin toàn ảnh thành 1 vector 2048 chiều. Đây là "ấn tượng chung" của CNN về cả bức ảnh
(không còn thông tin vị trí từng patch).

---

## Bên trong `cls_proj` — chỉ 1 lớp Linear

`cls_proj = nn.Linear(512 * expansion, tgt_dim)` = **`Linear(2048, 768)`**
([`resnet.py:571`](cakd_modified_files/resnet.py)).

```python
proj_token = self.cls_proj(cnn_token)    # (B, 2048) → (B, 768)
```

Cơ chế "2048 → 768" y hệt mọi Linear: nhân ma trận trọng số `W` (768×2048) với `cnn_token`,
cộng bias `b` (768):

$$
\boxed{\; t_S = W \cdot \texttt{cnn\_token} + b \;} \qquad (2048 \to 768)
$$

Mỗi trong 768 số đầu ra = 1 tổ hợp có trọng số của cả 2048 số đầu vào. `W, b` do model **tự học**
trong lúc train (qua `gl_loss`).

> **KHÁC `gl_proj`:** `cls_proj` chỉ là **1 Linear duy nhất**, KHÔNG chia nhóm (group-wise).
> Vì `cnn_token` là **1 vector tổng thể duy nhất** (đã gộp toàn ảnh, không còn 196 patch riêng lẻ)
> → không có gì để chia nhóm theo vị trí, nên 1 Linear là đủ.

---

## Vì sao 2048 và 768?

| Con số | Là gì | Bị ép bởi |
|---|---|---|
| **2048** (vào) | chiều `cnn_token` = kênh ra `layer4` (= 512 × expansion 4) | kiến trúc ResNet-50 |
| **768** (ra) | chiều class token của teacher ViT-B/16 | teacher ViT |

Giống câu chuyện của `f_S`: **2048 = "nơi student đi ra", 768 = "nơi phải đáp xuống để so với teacher"**.
Khác chỗ: `f_S` đi ra từ **1024** (layer3), còn `t_S` đi ra từ **2048** (layer4, sâu hơn) — vì
`cnn_token` lấy sau `layer4`.

---

## Vai trò của `t_S`

Dùng **1 chỗ duy nhất**: số hạng thứ 2 của `gl_loss` ([`dist_train_cakd.py:131`](dist_train_cakd.py)):

```python
gl_loss = mse(output,     tea_logits.detach())   \
        + mse(proj_token, tea_token)             \   # ← t_S khớp với class token teacher (hệ số 1)
        + 0.05 * mse(proj_feat, tea_feat.detach())
```

- **Ép `t_S` (tóm tắt toàn ảnh của student) giống `tea_token` (class token teacher ViT)** bằng MSE.
- Cả hai đều (B, 768) → khớp trực tiếp.
- Hệ số **1** (nặng như logits) — vì class token là "đại diện toàn ảnh" của ViT, tín hiệu rất quan trọng.
- ⚠️ **`tea_token` KHÔNG có `.detach()`** — đây là đầu ra teacher DUY NHẤT không detach trong toàn bộ
  loss. Nhưng teacher ở `eval()` và không nằm trong optimizer nên gradient qua `tea_token` cũng
  **không cập nhật gì** (xem FORMULAS.md, §3 loss gl).
- **KHÔNG** đưa vào GAN.

> So sánh với người anh em `f_S`: `gl_proj` sinh **feature theo từng patch** (196 token) để khớp
> `tea_feat`; còn `cls_proj` sinh **1 token tổng thể** để khớp `tea_token`. Hai cái = 2 mức độ:
> "chi tiết từng vùng" (f_S) và "tổng quan cả ảnh" (t_S).

---

## Bảng số chiều (cho `cls_proj`)

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| chiều vào (`cnn_token`) | 2048 (= 512 × expansion 4) | kênh ra `layer4` |
| chiều ra (`tgt_dim`) | 768 (= chiều ViT-B/16) | [`resnet.py:571`](cakd_modified_files/resnet.py) |
| kiểu lớp | `Linear(2048, 768)` — 1 cái, không chia nhóm | [`resnet.py:571`](cakd_modified_files/resnet.py) |
| khớp với teacher | `tea_token` = `x[:, 0]` (class token) | [`vision_transformer.py:364`](cakd_modified_files/vision_transformer.py) |

---

## Đủ bộ 4 đầu ra student — bảng tổng

| Đầu ra | Biến | Shape | Rẽ từ đâu | Qua lớp gì | Khớp teacher | Dùng trong |
|---|---|---|---|---|---|---|
| $z_S$ | `output` | (B,1000) | cnn_token | `fc` | `tea_logits` | cls_loss, gl_loss |
| $A^{qk}_S$ | `attn_weights[0]` | (B,196,196) | tmp (x_3) | `pca_proj` (QKᵀ) | `tea_attn_weights[2]` | pca_loss, **GAN** |
| $A^{vv}_S$ | `attn_weights[1]` | (B,196,196) | tmp (x_3) | `pca_proj` (VVᵀ) | `tea_attn_weights[3]` | pca_loss |
| $f_S$ | `proj_feat` | (B,196,768) | tmp (x_3) | `gl_proj` (16 nhóm) | `tea_feat` | gl_loss |
| **$t_S$** | **`proj_token`** | **(B,768)** | **cnn_token** | **`cls_proj` (1 Linear)** | **`tea_token`** | **gl_loss** |

Chú ý: 3 cái giữa rẽ từ **`tmp`** (nhánh distill, sau layer3), còn `z_S` và `t_S` rẽ từ **`cnn_token`**
(nhánh phân loại, sau layer4).

---

*Nguồn: `ResNet_CAKD._forward_impl` (dòng 631–666) và `self.cls_proj` (dòng 571)
trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py); `gl_loss` tại
[`dist_train_cakd.py:129-133`](dist_train_cakd.py).*
