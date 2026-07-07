# `Ā⁽¹⁾_T` (attention teacher) — từ ảnh gốc ra attention map qua những bước nào?

> Giải thích biến `tea_attn_weights[2][:, 1:, 1:]` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $\bar A^{(1)}_T$ | `tea_attn_weights[2][:, 1:, 1:]` | (B, 196, 196) | [`vision_transformer.py:209`](cakd_modified_files/vision_transformer.py) | attention **lớp cuối**, **SAU softmax**, bỏ CLS |
> | $\bar A^{(2)}_T$ | `tea_attn_weights[3][:, 1:, 1:]` | (B, 196, 196) | [`vision_transformer.py:209`](cakd_modified_files/vision_transformer.py) | attention **lớp cuối** (map thứ 2), bỏ CLS |

`Ā⁽¹⁾_T` là **ma trận attention (B, 196, 196)** của teacher ViT — "patch nào chú ý patch nào" ở
**lớp Transformer cuối cùng**. Đây là **"đáp án mẫu attention"** để student bắt chước (`A^{qk}_S` khớp
với `Ā⁽¹⁾_T`). Nó cũng là **"mẫu THẬT"** đưa vào discriminator (GAN).

> File này kèm luôn `Ā⁽²⁾_T` (`tea_attn_weights[3]`) vì nó ra cùng chỗ, chỉ khác là map thứ 2 của lớp cuối.

---

## Hành trình `image → tea_attn_weights`

```
image                          (B, 3, 224, 224)
  │
  ├─ _process_input             → (B, 196, 768)     cắt patch 16×16 + nhúng (xem file logits teacher)
  ├─ cat[class_token, .]        → (B, 197, 768)     thêm class token vào đầu
  │
  │   ════════ ENCODER: 12 lớp Transformer (Encoder.forward) ════════
  ├─ + pos_embedding                                        [195]  cộng nhãn vị trí
  ├─ lớp 0 .. lớp 9   (10 lớp đầu)  → chỉ lấy output        [201]  KHÔNG lấy attention
  ├─ lớp 10 (áp chót): x, attn_weights_2 = layer(x, True)   [203]  lấy attention (2 map)
  ├─ lớp 11 (cuối):    x, attn_weights_1 = layer(x, True)   [205]  lấy attention (2 map)  ★
  │
  └─ return ln(x), [ attn_weights_2[0], attn_weights_2[1],       [209]
                     attn_weights_1[0], attn_weights_1[1] ]
                     └── index 0 ──┘ └── index 1 ──┘ └ index 2 ┘ └ index 3 ┘
                        (lớp áp chót)                  (lớp CUỐI) ★

  → tea_attn_weights = list 4 phần tử.  Code CHỈ dùng [2] và [3] (lớp cuối).
     tea_attn_weights[2] = attn_weights_1[0]  → Ā⁽¹⁾_T   (shape (B, 197, 197))
     tea_attn_weights[3] = attn_weights_1[1]  → Ā⁽²⁾_T
```

Cuối cùng, ở file train ([`dist_train_cakd.py:124`](dist_train_cakd.py)), cắt bỏ class token:

```
tea_attn_weights[2][:, 1:, 1:]   → (B, 196, 196) = Ā⁽¹⁾_T   (bỏ hàng & cột của class token)
```

---

## Giải thích từng ý

### 1) Attention sinh ở đâu? — bên trong `EncoderBlock` ([`vision_transformer.py:142-156`](cakd_modified_files/vision_transformer.py))

Mỗi lớp Transformer có 1 khối self-attention:

```python
self.self_attention = nn.MultiheadAttention(768, num_heads=12, batch_first=True)   # [135]
...
x, attn_weights = self.self_attention(query=x, key=x, value=x, need_weights=need_weights)   # [148]
```

- `query=key=value=x` → mỗi token nhìn **tất cả** token khác (self-attention).
- `need_weights=True` → hàm trả về **CẢ output VÀ ma trận attention** `attn_weights` (nếu `False` chỉ trả output). Đây là lý do 10 lớp đầu không tốn công lấy attention, chỉ 2 lớp cuối mới bật cờ.
- `attn_weights` = "ai chú ý ai" giữa 197 token → ma trận **(B, 197, 197)**, **ĐÃ qua softmax** (MultiheadAttention softmax bên trong), và đã **trung bình 12 head**.

Cơ chế bên trong giống hệt phần student đã học: $\frac{QK^\top}{\sqrt d}$ → **softmax** → trọng số attention. Khác biệt: teacher **trả ra trọng số SAU softmax** (xác suất), còn student trả `dots_qk` **TRƯỚC softmax**.

### 2) Vì sao list có 4 phần tử? — `Encoder.forward` ([`vision_transformer.py:199-209`](cakd_modified_files/vision_transformer.py))

Encoder chạy 12 lớp, chỉ **2 lớp cuối** bật `need_weights=True`:

| Biến | Là gì | Vào list ở index |
|---|---|---|
| `attn_weights_2` | attention lớp **áp chót** (lớp 10) | `[0]`, `[1]` |
| `attn_weights_1` | attention lớp **cuối** (lớp 11) | `[2]`, `[3]` |

Mỗi lớp cho **2 ma trận** (map `[0]` và map `[1]`) → 2 lớp × 2 = **4 phần tử**:

```python
return self.ln(x), [attn_weights_2[0], attn_weights_2[1], attn_weights_1[0], attn_weights_1[1]]
#                   └─ index 0 ─┘ └─ 1 ─┘  └─ index 2 ─┘ └─ index 3 ─┘
```

**Code CAKD chỉ dùng `[2]` và `[3]` — tức attention của LỚP CUỐI** (lớp sâu nhất, "chín" nhất).
Hai map `[0]`, `[1]` của lớp áp chót được trả ra nhưng **không dùng** trong loss.

### 3) `[2]` và `[3]` khác nhau chỗ nào, ghép với ai?

Hai map của lớp cuối được **ghép cặp với 2 attention của student** trong `pca_loss`
([`dist_train_cakd.py:123-127`](dist_train_cakd.py)):

```python
pca_loss = 0.2  * mse(attn_weights[0]=A^qk_S, tea_attn_weights[2][:,1:,1:])   # Ā⁽¹⁾_T ghép QK student
         + 0.05 * mse(attn_weights[1]=A^vv_S, tea_attn_weights[3][:,1:,1:])   # Ā⁽²⁾_T ghép VV student
```

- `tea_attn_weights[2]` = **Ā⁽¹⁾_T** → ghép với `A^{qk}_S` (attention query-key của student).
- `tea_attn_weights[3]` = **Ā⁽²⁾_T** → ghép với `A^{vv}_S` (attention value-value của student).

### 4) `[:, 1:, 1:]` — bỏ class token

Attention teacher là (B, **197**, 197) vì có class token ở vị trí 0. Nhưng student chỉ có **196**
patch (không có class token trong attention). Để 2 bên khớp shape:

```
tea_attn_weights[2]          (B, 197, 197)
        │  [:, 1:, 1:]   ← bỏ HÀNG 0 và CỘT 0 (của class token)
        ▼
Ā⁽¹⁾_T                       (B, 196, 196)     ← chỉ còn quan hệ giữa 196 patch
```

`1:` = "lấy từ index 1 trở đi" → cắt bỏ dòng/cột đầu (class token), giữ 196 patch thật.

---

## Ý nghĩa & vai trò của `Ā⁽¹⁾_T`

`Ā⁽¹⁾_T` dùng ở **2 chỗ** (đây là điểm đặc biệt — attention teacher là tín hiệu "đắt giá" nhất):

**① Trong `pca_loss`** — làm "đáp án" để student bắt chước ([`dist_train_cakd.py:123`](dist_train_cakd.py)):
```python
0.2 * mse(A^qk_S, tea_attn_weights[2][:,1:,1:].detach())   # ép attention student giống teacher
```

**② Trong GAN** — làm "mẫu THẬT" cho discriminator ([`dist_train_cakd.py:106`](dist_train_cakd.py)):
```python
input_d_real = tea_attn_weights[2][:, 1:, 1:].clone()[:, None, :, :].detach()   # (B, 1, 196, 196)
pred_real = discriminator(input_d_real)   # D chấm điểm "đây là attention THẬT"
```
`[:, None, :, :]` chèn 1 chiều kênh → (B, 1, 196, 196) cho hợp đầu vào Conv2d của discriminator.
Student thì cố làm `A^{qk}_S` "trông giống thật" để đánh lừa D (số hạng generator).

> ⚠️ **Bất đối xứng softmax:** `Ā⁽¹⁾_T` là xác suất **SAU softmax** (0..1, mỗi hàng tổng =1), còn
> student `A^{qk}_S` là điểm **TRƯỚC softmax**. `pca_loss` (MSE) ép 2 thứ khác thang đo — xem
> FORMULAS.md mục 6 điểm 7.

---

## Bảng đối chiếu student ↔ teacher (attention)

| | Student `A^{qk}_S` | Teacher `Ā⁽¹⁾_T` |
|---|---|---|
| Sinh từ | `pca_proj` gắn trên feature layer3 | lớp Transformer cuối của ViT |
| Trước/sau softmax | **TRƯỚC** (`dots_qk` thô) | **SAU** (đã softmax) |
| Trung bình head | 16 head | 12 head |
| Shape gốc | (B, 196, 196) | (B, 197, 197) → cắt còn (B, 196, 196) |
| Class token | không có | có (bị `[:,1:,1:]` cắt bỏ) |
| Vai trò | mẫu "GIẢ" cho GAN + bị ép giống teacher | mẫu "THẬT" cho GAN + làm đáp án |

---

## Bảng chỉ số `tea_attn_weights` (list 4 phần tử)

| Index | = | Lớp | Dùng trong code? | Ghép với |
|---|---|---|---|---|
| `[0]` | `attn_weights_2[0]` | áp chót (10) | ❌ không | — |
| `[1]` | `attn_weights_2[1]` | áp chót (10) | ❌ không | — |
| **`[2]`** | `attn_weights_1[0]` | **cuối (11)** | ✅ **Ā⁽¹⁾_T** | `A^{qk}_S` (pca_loss + GAN) |
| **`[3]`** | `attn_weights_1[1]` | **cuối (11)** | ✅ **Ā⁽²⁾_T** | `A^{vv}_S` (pca_loss) |

---

*Nguồn: `Encoder.forward` (dòng 193–209), `EncoderBlock.forward` (dòng 142–156)
trong [`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py);
`pca_loss` + chuẩn bị GAN tại [`dist_train_cakd.py:106-127`](dist_train_cakd.py).*
