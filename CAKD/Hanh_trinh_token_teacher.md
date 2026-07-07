# `t_T` (class token teacher) — từ ảnh gốc ra token qua những bước nào?

> Giải thích biến `tea_token` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $t_T$ | `tea_token` | (B, 768) | [`vision_transformer.py:364`](cakd_modified_files/vision_transformer.py) `cls_token = x[:, 0]` | class token, **KHÔNG** detach |

`t_T` là **class token của teacher ViT**: 1 vector 768 chiều cho mỗi ảnh (shape (B, 768)) —
**"bản tóm tắt toàn ảnh"** của teacher. Đây là "đáp án mẫu" để student khớp bằng `t_S` (đầu ra
`cls_proj`). Nó chính là **token vị trí 0** ở đầu ra encoder — token đặc biệt được thêm vào từ đầu
để hút thông tin toàn ảnh.

---

## Hành trình `image → t_T`

```
image                          (B, 3, 224, 224)
  │
  ├─ _process_input             → (B, 196, 768)     cắt patch 16×16 + nhúng
  ├─ cat[class_token, .]        → (B, 197, 768)     thêm class_token (tham số học được) vào vị trí 0
  │
  ├─ encoder(x)                 → x (B, 197, 768)   [361]  12 lớp attention: class token "hút" thông tin
  │                                                        từ toàn bộ 196 patch
  │
  └─ cls_token = x[:, 0]        → (B, 768)          [364]  = tea_token = t_T   ★
                                                            lấy RIÊNG token vị trí 0
```

`t_T` chỉ là **1 phép cắt** `x[:, 0]` — không qua thêm lớp nào. (Sau đó `heads(cls_token)` mới ra
logits `z_T`, nhưng đó là bước riêng cho nhánh phân loại.)

---

## Giải thích từng ý

### 1) `class_token` từ đâu ra? ([`vision_transformer.py:280`](cakd_modified_files/vision_transformer.py))

```python
self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))   # 1 token 768 chiều, HỌC ĐƯỢC
```

- `class_token` **không phải lấy từ ảnh** — nó là **tham số của model** (khởi tạo bằng 0, rồi học dần khi train ViT gốc). Giống 1 "ô trống thông minh" chờ được điền thông tin.
- Ở [`vision_transformer.py:357-358`](cakd_modified_files/vision_transformer.py), nó được nhân bản cho cả batch rồi **gắn vào đầu** chuỗi 196 patch → thành 197 token.

### 2) Vì sao lấy `x[:, 0]` làm "tóm tắt toàn ảnh"?

Trong self-attention, **mọi token nhìn mọi token**. Qua 12 lớp, class token (vị trí 0) liên tục
"hút" thông tin từ tất cả 196 patch → cuối cùng nó **cô đọng đặc trưng của cả bức ảnh** vào 1 vector.

```
x (B, 197, 768) sau encoder
   │  x[:, 0]  ← nhặt riêng token vị trí 0 (class token)
   ▼
t_T (B, 768)    = "cả bức ảnh gói trong 768 số"
```

Đây là lý do ViT dùng **chính class token** (không phải trung bình các patch) để phân loại — nó
được thiết kế để làm đại diện toàn ảnh.

### 3) So với student

Student **không có** class token kiểu ViT. Thay vào đó student dùng `cnn_token` (vector 2048 chiều
sau avgpool) rồi chiếu qua `cls_proj` (2048→768) để tạo `t_S` "đóng vai" class token → so với `t_T`.

---

## Ý nghĩa & vai trò của `t_T`

Dùng **1 chỗ duy nhất**: số hạng thứ 2 của `gl_loss` ([`dist_train_cakd.py:131`](dist_train_cakd.py)):

```python
gl_loss = mse(output, tea_logits.detach())   \
        + mse(proj_token, tea_token)         \   # ← t_S khớp với t_T (hệ số 1)
        + 0.05 * mse(proj_feat, tea_feat.detach())
```

- **Ép `t_S` (tóm tắt toàn ảnh student) giống `t_T` (class token teacher)** bằng MSE.
- Cả hai (B, 768) → khớp trực tiếp.
- Hệ số **1** (nặng) — class token là đại diện toàn ảnh, tín hiệu rất quan trọng.
- **KHÔNG** đưa vào GAN.

### ⚠️ Điểm đặc biệt: `t_T` KHÔNG có `.detach()`

Đây là **đầu ra teacher DUY NHẤT** không `.detach()` khi dùng (so với `tea_logits.detach()`,
`tea_feat.detach()`, và attention teacher cũng detach):

```python
mse(proj_token, tea_token)   # tea_token KHÔNG detach ← khác 3 tín hiệu teacher kia
```

**Nhưng thực tế vẫn vô hại:** teacher chạy `eval()` và **không nằm trong optimizer nào**
([`dist_train_cakd.py:66`](dist_train_cakd.py)) → dù gradient có chảy ngược qua `tea_token`,
**không trọng số teacher nào được cập nhật**. Nên kết quả giống như có detach; chỉ tốn thêm chút
bộ nhớ/tính toán cho gradient thừa. (Xem FORMULAS.md §3 loss gl.)

---

## Cặp đôi `t_S` ↔ `t_T`

```
STUDENT                                       TEACHER
cnn_token (B,2048) sau avgpool                x[:, 0] sau encoder (class token)
      │  cls_proj (Linear 2048→768)                 │
      ▼                                             ▼
   t_S (B,768)  ───── MSE (gl_loss, hệ số 1) ─────  t_T (B,768)
```

Student ép "tóm tắt toàn ảnh kiểu CNN" (`t_S`) tiến sát "tóm tắt toàn ảnh kiểu ViT" (`t_T`).

---

## Bảng: 3 đầu ra teacher tách từ `x` (B,197,768)

| Đầu ra | Lấy thế nào | Shape | detach? | Khớp student |
|---|---|---|---|---|
| **`tea_token` ($t_T$)** | **`x[:, 0]`** | **(B, 768)** | **❌ KHÔNG** | **`t_S` (cls_proj)** |
| `tea_feat` ($f_T$) | `x[:, 1:]` | (B, 196, 768) | ✅ có | `f_S` (gl_proj) |
| `tea_logits` ($z_T$) | `heads(x[:, 0])` | (B, 1000) | ✅ có | `z_S` (fc) |

---

## 🎉 Trọn bộ 8 tín hiệu CAKD (4 student + 4 teacher)

| # | Student | ↔ khớp ↔ | Teacher | Qua loss |
|---|---|---|---|---|
| 1 | `output` ($z_S$) | ↔ | `tea_logits` ($z_T$) | cls_loss + gl_loss |
| 2 | `attn_weights[0]` ($A^{qk}_S$) | ↔ | `tea_attn_weights[2]` ($\bar A^{(1)}_T$) | pca_loss + **GAN** |
| 3 | `attn_weights[1]` ($A^{vv}_S$) | ↔ | `tea_attn_weights[3]` ($\bar A^{(2)}_T$) | pca_loss |
| 4 | `proj_feat` ($f_S$) | ↔ | `tea_feat` ($f_T$) | gl_loss |
| 5 | `proj_token` ($t_S$) | ↔ | `tea_token` ($t_T$) | gl_loss |

*(4 đầu ra student vì `attn_weights` là list 2 phần tử → tính là 2 tín hiệu; teacher tương tự.)*

---

*Nguồn: `VisionTransformer.forward` (dòng 351–374), `class_token` (dòng 280) trong
[`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py);
`gl_loss` tại [`dist_train_cakd.py:129-133`](dist_train_cakd.py).*
