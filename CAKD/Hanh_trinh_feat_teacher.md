# `f_T` (patch feature teacher) — từ ảnh gốc ra feature qua những bước nào?

> Giải thích biến `tea_feat` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $f_T$ | `tea_feat` | (B, 196, 768) | [`vision_transformer.py:365`](cakd_modified_files/vision_transformer.py) `feats = x[:, 1:]` | 196 patch token, `.detach()` khi dùng |

`f_T` là **feature theo từng patch của teacher ViT**: **196 token, mỗi token 768 chiều**
(shape (B, 196, 768)). Đây là "đáp án mẫu feature" mà student khớp bằng `f_S` (đầu ra `gl_proj`).
Nó chính là **196 patch token ở đầu ra cuối cùng của encoder** — chỉ bỏ đi class token.

---

## Hành trình `image → f_T`

```
image                          (B, 3, 224, 224)
  │
  ├─ _process_input             → (B, 196, 768)     cắt patch 16×16 + nhúng
  ├─ cat[class_token, .]        → (B, 197, 768)     thêm class token vào ĐẦU (vị trí 0)
  │
  ├─ encoder(x)                 → x (B, 197, 768)   [361]  qua 12 lớp attention + LayerNorm cuối
  │                                                        (mỗi token đã "trộn" thông tin toàn ảnh)
  │
  └─ feats = x[:, 1:]           → (B, 196, 768)     [365]  = tea_feat = f_T   ★
                                                            bỏ token vị trí 0 (class token), giữ 196 patch
```

So với `tea_logits` (đi tiếp qua `heads`) và `tea_token` (lấy `x[:, 0]`), `f_T` chỉ đơn giản là
**lấy 196 token còn lại** (`x[:, 1:]`) — không qua thêm lớp nào nữa.

---

## Giải thích từng ý

### 1) `x` sau encoder là gì?

Sau khi qua **12 lớp Transformer + LayerNorm cuối** ([`vision_transformer.py:361`](cakd_modified_files/vision_transformer.py)), `x` có shape (B, **197**, 768):

```
197 token = 1 (class token, vị trí 0)  +  196 (patch token, vị trí 1..196)
```

Mỗi token giờ **không còn là patch "thô"** như lúc mới cắt ảnh — nó đã qua 12 vòng self-attention,
"hút" thông tin từ mọi token khác → trở thành **đặc trưng ngữ cảnh phong phú** của vùng ảnh đó.

### 2) `feats = x[:, 1:]` — bỏ class token, giữ 196 patch

```python
cls_token = x[:, 0]    # (B, 768)      token vị trí 0  → dùng phân loại (t_T)
feats     = x[:, 1:]   # (B, 196, 768) token vị trí 1..196 → f_T
```

- `x[:, 1:]` = "lấy từ token index 1 đến hết" → bỏ token 0 (class token), giữ lại **196 patch token**.
- Đây là **đặc trưng chi tiết theo từng vùng ảnh** của teacher (khác `cls_token` là tóm tắt toàn ảnh).

> Cách chia `x` thành `cls_token` (vị trí 0) và `feats` (vị trí 1:) là điểm mấu chốt: 1 token lo
> "tổng quan", 196 token lo "chi tiết". Student cũng khớp cả 2 mức: `t_S`↔`t_T` và `f_S`↔`f_T`.

---

## Ý nghĩa & vai trò của `f_T`

Dùng **1 chỗ duy nhất**: số hạng thứ 3 của `gl_loss` ([`dist_train_cakd.py:132`](dist_train_cakd.py)):

```python
gl_loss = mse(output, tea_logits.detach())       \
        + mse(proj_token, tea_token)             \
        + 0.05 * mse(proj_feat, tea_feat.detach())   # ← f_S khớp với f_T (patch feature)
                        #  └ f_S      └ f_T
```

- **Ép `f_S` (patch feature student, qua `gl_proj`) giống `f_T`** bằng MSE, từng patch từng chiều.
- Cả hai đều (B, 196, 768) → khớp 1-1.
- Hệ số **0.05** (nhẹ) — feature theo patch là tín hiệu phụ, không nặng bằng logits/token (hệ số 1).
- `.detach()` → teacher không bị cập nhật.
- **KHÔNG** đưa vào GAN.

---

## Cặp đôi `f_S` ↔ `f_T` (student học feature teacher)

```
STUDENT                                    TEACHER
x_3 (14×14 CNN) → tmp (196,1024)           patch (196,768) → 12 lớp attention
      │  gl_proj (16 nhóm, 1024→768)              │  x[:, 1:]
      ▼                                           ▼
   f_S (B,196,768)  ───── MSE (gl_loss, 0.05) ───  f_T (B,196,768)
```

`gl_proj` bên student tồn tại **chính là để** biến feature CNN (1024 chiều) sang đúng dạng `f_T`
(768 chiều) mà so được. `f_T` là "chuẩn" mà `gl_proj` phải học bắt chước.

---

## Bảng đối chiếu 3 đầu ra teacher tách từ `x`

Cả 3 đều lấy từ cùng 1 tensor `x` (B, 197, 768) sau encoder:

| Đầu ra | Lấy thế nào | Shape | Ý nghĩa | Khớp với student |
|---|---|---|---|---|
| `tea_token` ($t_T$) | `x[:, 0]` | (B, 768) | class token — tóm tắt toàn ảnh | `t_S` (cls_proj) |
| **`tea_feat` ($f_T$)** | **`x[:, 1:]`** | **(B, 196, 768)** | **196 patch — chi tiết từng vùng** | **`f_S` (gl_proj)** |
| `tea_logits` ($z_T$) | `heads(x[:, 0])` | (B, 1000) | logits phân loại | `z_S` (fc) |

---

*Nguồn: `VisionTransformer.forward` (dòng 351–374) trong
[`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py);
`gl_loss` tại [`dist_train_cakd.py:129-133`](dist_train_cakd.py).*
