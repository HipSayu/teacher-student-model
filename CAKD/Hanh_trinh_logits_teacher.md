# `z_T` (logits teacher) — từ ảnh gốc ra logits qua những bước nào?

> Giải thích biến `tea_logits` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $z_T$ | `tea_logits` | (B, 1000) | [`vision_transformer.py:367`](cakd_modified_files/vision_transformer.py) | logits teacher, `.detach()` khi dùng |

`z_T` là **vector 1000 logits** của **teacher ViT-B/16** — giống vai trò `z_S` bên student, nhưng do
**thầy giáo** (model lớn, đã pretrain, ĐÓNG BĂNG) sinh ra. Nó là "đáp án mềm" để student bắt chước.
Đường đi hoàn toàn khác student: teacher là **Transformer thuần**, không có CNN.

> ⚙️ Teacher chạy ở `eval()` và **không nằm trong optimizer** ([`dist_train_cakd.py:66`](dist_train_cakd.py))
> → mọi đầu ra teacher chỉ là "tín hiệu cố định", khi dùng đều `.detach()` (cắt gradient).

---

## Hành trình `image → z_T`

```
image                          (B,   3, 224, 224)   ← ảnh RGB đầu vào
  │
  │   ════════ 1) CẮT PATCH + NHÚNG (_process_input) ════════
  ├─ conv_proj (Conv2d 16×16, stride 16) → (B, 768, 14, 14)   [341]  mỗi ô 16×16 → 1 vector 768
  ├─ reshape                             → (B, 768, 196)       [343]  duỗi lưới 14×14 = 196
  ├─ permute(0, 2, 1)                    → (B, 196, 768)       [347]  = 196 token, mỗi token 768 chiều
  │
  │   ════════ 2) GẮN CLASS TOKEN ════════
  ├─ cat([class_token, x], dim=1)        → (B, 197, 768)       [358]  thêm 1 token "đại diện toàn ảnh" vào đầu
  │
  │   ════════ 3) QUA ENCODER (12 lớp attention) ════════
  ├─ encoder(x)   (+ pos_embedding, 12× TransformerBlock, LayerNorm cuối)  [361]
  │     → x (B, 197, 768)   +   attn_weights (list 4)
  │
  │   ════════ 4) TÁCH & PHÂN LOẠI ════════
  ├─ cls_token = x[:, 0]                 → (B, 768)            [364]  lấy RIÊNG token vị trí 0
  └─ heads(cls_token)  (Linear 768→1000) → (B, 1000)          [367]  = tea_logits = z_T   ★ LOGITS
```

---

## Giải thích 4 giai đoạn

### 1) Cắt patch + nhúng — `_process_input` ([`vision_transformer.py:331-349`](cakd_modified_files/vision_transformer.py))

ViT không có conv trích đặc trưng dần như CNN. Nó **cắt thẳng ảnh thành các ô 16×16** rồi nhúng mỗi ô thành 1 token:

```python
self.conv_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)   # [271-273]
```

- `kernel=stride=16` ⇒ conv này **không chồng lấn**: nó chia ảnh 224×224 thành lưới 14×14 ô, **mỗi ô 16×16 pixel → nén thành 1 vector 768 chiều**. Đây chính là "patch embedding".
- 224/16 = 14 → **14×14 = 196 patch**.
- `reshape` + `permute` đưa về dạng chuỗi **(B, 196, 768)**: 196 token, mỗi token 768 chiều.

> So với student: student **duỗi feature map layer3** (đã qua nhiều conv) thành 196 token;
> teacher **cắt thẳng ảnh gốc** thành 196 patch. Cùng ra 196 token nhưng cách tạo khác hẳn.

### 2) Gắn class token ([`vision_transformer.py:356-358`](cakd_modified_files/vision_transformer.py))

```python
batch_class_token = self.class_token.expand(n, -1, -1)   # nhân bản cho cả batch
x = torch.cat([batch_class_token, x], dim=1)             # (B, 196, 768) → (B, 197, 768)
```

- `class_token` là **1 token học được** (tham số, khởi tạo 0 — [`vision_transformer.py:280`](cakd_modified_files/vision_transformer.py)), gắn vào **đầu** chuỗi.
- Sau khi qua encoder, token này **hút thông tin từ toàn bộ 196 patch** → trở thành "bản tóm tắt toàn ảnh", dùng để phân loại. Giờ chuỗi dài **197 = 1 (class) + 196 (patch)**.

### 3) Qua Encoder ([`vision_transformer.py:193-209`](cakd_modified_files/vision_transformer.py))

```python
x, attn_weights = self.encoder(x)   # [361]
```

Encoder = cộng **position embedding** (báo cho model biết token nào ở vị trí nào) → chạy qua **12 lớp Transformer** (mỗi lớp = self-attention + MLP) → **LayerNorm** cuối. Đầu ra vẫn (B, 197, 768) nhưng mỗi token đã được "trộn thông tin" với mọi token khác.

*(Encoder cũng nhả ra `attn_weights` = 4 attention map của 2 lớp cuối — đó là `tea_attn_weights`,
xem file riêng. Ở đây ta chỉ quan tâm nhánh sinh logits.)*

### 4) Tách class token & phân loại ([`vision_transformer.py:364-367`](cakd_modified_files/vision_transformer.py))

```python
cls_token = x[:, 0]        # (B, 768)   ← chỉ lấy token vị trí 0 (class token)
x = self.heads(cls_token)  # (B, 1000)  ← Linear(768 → 1000) = tea_logits
```

- `x[:, 0]` = nhặt riêng **class token** (vị trí 0) ra khỏi 197 token → vector 768 chiều đại diện cả ảnh. (Cái này chính là `tea_token` = $t_T$, xem file riêng.)
- `heads` = `Linear(768, 1000)` ([`vision_transformer.py:298`](cakd_modified_files/vision_transformer.py)) → biến 768 chiều thành **1000 điểm số lớp** = `tea_logits`.

---

## Ý nghĩa & vai trò của `z_T`

`z_T` = 1000 logits của teacher — "thầy giáo nghĩ ảnh này thuộc lớp nào, với độ tự tin ra sao".
Dùng **1 chỗ**: số hạng thứ 1 của `gl_loss` ([`dist_train_cakd.py:130`](dist_train_cakd.py)):

```python
gl_loss = mse(output, tea_logits.detach()) + ...
#             └ z_S     └ z_T (detach)
```

- **Ép logits student `z_S` giống logits teacher `z_T`** bằng MSE → đây là **knowledge distillation "mềm"**: student không chỉ học nhãn đúng (cls_loss), mà còn học **cả phân bố dự đoán** của teacher (lớp nào teacher thấy "hơi giống" cũng truyền lại).
- `.detach()` → teacher không bị cập nhật.
- Hệ số **1** (nặng).

> ⚠️ **Khác bài báo:** ở đây khớp logits bằng **MSE**, không phải KL-divergence + temperature như KD kinh điển (xem FORMULAS.md, mục 6, điểm 3).

---

## Bảng số chiều (ViT-B/16, teacher)

| Đại lượng | Giá trị | Ghi chú |
|---|---|---|
| `patch_size` | 16 | ô 16×16 pixel |
| số patch | 196 (= 14×14) | 224/16 = 14 |
| `hidden_dim` | 768 | chiều mỗi token |
| chuỗi sau class token | 197 (= 196 + 1) | thêm class token |
| `num_layers` | 12 | số lớp Transformer |
| `num_heads` | 12 | số head mỗi lớp |
| `heads` (phân loại) | `Linear(768, 1000)` | → logits |

---

## Đối chiếu student ↔ teacher (cùng ra logits 1000)

| | Student (`z_S`) | Teacher (`z_T`) |
|---|---|---|
| Kiến trúc | CNN (ResNet-50) | Transformer (ViT-B/16) |
| Tạo token | duỗi feature map layer3 (14×14) | cắt thẳng ảnh thành patch 16×16 |
| Ra logits qua | avgpool → `fc` (2048→1000) | class token → `heads` (768→1000) |
| Có học không | CÓ (đang được dạy) | KHÔNG (đóng băng, `eval`) |

---

*Nguồn: `VisionTransformer.forward` (dòng 351–374), `_process_input` (dòng 331–349),
`Encoder.forward` (dòng 193–209) trong [`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py);
`gl_loss` tại [`dist_train_cakd.py:129-133`](dist_train_cakd.py).*
