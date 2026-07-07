# `Ā⁽²⁾_T` (attention teacher — map thứ 2) — từ ảnh gốc ra qua những bước nào?

> Giải thích biến `tea_attn_weights[3][:, 1:, 1:]` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $\bar A^{(2)}_T$ | `tea_attn_weights[3][:, 1:, 1:]` | (B, 196, 196) | [`vision_transformer.py:209`](cakd_modified_files/vision_transformer.py) | attention **lớp cuối** (map thứ 2), bỏ CLS |

`Ā⁽²⁾_T` là **map attention thứ 2 của lớp Transformer cuối** trong teacher ViT. Nó ra **cùng một chỗ**
với `Ā⁽¹⁾_T` (`tea_attn_weights[2]`), chỉ khác **index `[3]` thay vì `[2]`** — tức lấy phần tử thứ 2
mà lớp cuối trả ra. Vai trò: **ghép với `A^{vv}_S` của student** (nhánh value-value).

> 👉 File này gần như trùng [`Hanh_trinh_attn_teacher.md`](Hanh_trinh_attn_teacher.md).
> Nếu đã đọc file đó, chỉ cần xem mục **"Khác biệt so với `Ā⁽¹⁾_T`"** ở cuối.

---

## Hành trình `image → Ā⁽²⁾_T`

```
image                          (B, 3, 224, 224)
  │
  ├─ _process_input             → (B, 196, 768)     cắt patch 16×16 + nhúng
  ├─ cat[class_token, .]        → (B, 197, 768)     thêm class token
  │
  │   ════════ ENCODER: 12 lớp Transformer ════════
  ├─ + pos_embedding                                        [195]
  ├─ lớp 0 .. lớp 9                → chỉ output              [201]
  ├─ lớp 10 (áp chót)             → attn_weights_2 (2 map)  [203]  không dùng
  ├─ lớp 11 (CUỐI)               → attn_weights_1 (2 map)  [205]  ★
  │
  └─ return ln(x), [attn_weights_2[0], attn_weights_2[1],
                    attn_weights_1[0], attn_weights_1[1]]        [209]
                    └ [0] ┘ └ [1] ┘  └ [2] ┘  └ [3] ┘ ★

  → tea_attn_weights[3] = attn_weights_1[1]  → Ā⁽²⁾_T   (shape (B, 197, 197))
```

Ở file train ([`dist_train_cakd.py:126`](dist_train_cakd.py)), cắt bỏ class token:

```
tea_attn_weights[3][:, 1:, 1:]   → (B, 196, 196) = Ā⁽²⁾_T
```

---

## Khác biệt so với `Ā⁽¹⁾_T`

Mọi bước tạo ra **hoàn toàn giống** `Ā⁽¹⁾_T` (cùng lớp cuối, cùng đã-softmax, cùng bỏ CLS bằng
`[:,1:,1:]`). Chỉ khác **3 điểm**:

| | `Ā⁽¹⁾_T` (`tea_attn_weights[2]`) | `Ā⁽²⁾_T` (`tea_attn_weights[3]`) |
|---|---|---|
| Index | `[2]` = `attn_weights_1[0]` (map thứ 1) | `[3]` = `attn_weights_1[1]` (map thứ 2) |
| Ghép với student | `A^{qk}_S` (query-key) | `A^{vv}_S` (value-value) |
| Trọng số trong `pca_loss` | **0.2** | **0.05** (nhẹ hơn) |
| Đưa vào GAN? | ✅ CÓ (làm "mẫu THẬT") | ❌ KHÔNG |

Cả hai đều: lớp cuối · đã softmax · trung bình 12 head · (B,197,197) → cắt còn (B,196,196).

---

## Vai trò của `Ā⁽²⁾_T`

Dùng **1 chỗ duy nhất**: số hạng thứ 2 của `pca_loss` ([`dist_train_cakd.py:125-127`](dist_train_cakd.py)):

```python
pca_loss = 0.2  * mse(attn_weights[0]=A^qk_S, tea_attn_weights[2][:,1:,1:].detach()) \
         + 0.05 * mse(attn_weights[1]=A^vv_S, tea_attn_weights[3][:,1:,1:].detach())
                       #        ↑↑↑ A^vv_S ghép với Ā⁽²⁾_T (map thứ 2 lớp cuối)
```

- **Ép `A^{vv}_S` (attention value-value student) giống `Ā⁽²⁾_T`** bằng MSE.
- Hệ số **0.05** (nhẹ) — đây là tín hiệu distill phụ, không nặng bằng `Ā⁽¹⁾_T` (0.2).
- `.detach()` → teacher không bị cập nhật.
- **KHÔNG** đưa vào discriminator (chỉ `Ā⁽¹⁾_T` mới làm "mẫu thật" cho GAN).

> ⚠️ **Bất đối xứng softmax:** giống `Ā⁽¹⁾_T`, `Ā⁽²⁾_T` là xác suất **SAU softmax**, còn student
> `A^{vv}_S` là điểm **TRƯỚC softmax**.

---

## Bảng chỉ số `tea_attn_weights` (nhắc lại)

| Index | = | Lớp | Dùng? | Ghép với |
|---|---|---|---|---|
| `[0]` | `attn_weights_2[0]` | áp chót | ❌ | — |
| `[1]` | `attn_weights_2[1]` | áp chót | ❌ | — |
| `[2]` | `attn_weights_1[0]` | cuối | ✅ **Ā⁽¹⁾_T** | `A^{qk}_S` (pca_loss + GAN) |
| **`[3]`** | **`attn_weights_1[1]`** | **cuối** | ✅ **Ā⁽²⁾_T** | **`A^{vv}_S`** (pca_loss) |

---

*Nguồn: `Encoder.forward` (dòng 193–209) trong
[`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py);
`pca_loss` tại [`dist_train_cakd.py:123-127`](dist_train_cakd.py). Chi tiết đầy đủ về cơ chế
attention xem [`Hanh_trinh_attn_teacher.md`](Hanh_trinh_attn_teacher.md).*
