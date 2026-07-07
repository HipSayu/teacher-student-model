# `A^{vv}_S` (attention VV của student) — từ ảnh gốc ra attention map qua những bước nào?

> Giải thích biến `attn_weights[1]` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $A^{vv}_S$ | `attn_weights[1]` | (B, 196, 196) | [`resnet.py:648,651`](cakd_modified_files/resnet.py) | $\frac{1}{H}\sum_h \frac{V_S V_S^\top}{\sqrt{d_h}}$ — **TRƯỚC softmax** |

`A^{vv}_S` là **ma trận (B, 196, 196)** đo "value của các ô lưới **giống nhau** tới mức nào".
Nó là **tín hiệu distill thứ 2** — sinh ra ở **cùng một chỗ** với $A^{qk}_S$ (khối `pca_proj`),
chỉ khác **công thức tính**: dùng **V với chính V** thay vì **Q với K**.

> 👉 File này gần như trùng [`Hanh_trinh_attn_qk_student.md`](Hanh_trinh_attn_qk_student.md).
> Nếu đã đọc file đó, chỉ cần xem mục **"Khác biệt duy nhất so với `A^{qk}_S`"** ở cuối.

---

## Hành trình `image → A^{vv}_S`

```
image                          (B,   3, 224, 224)   ← ảnh RGB đầu vào
  │
  ├─ stem (conv1+bn1+relu+maxpool) → (B,  64,  56,  56)   [633-636]
  ├─ layer1                        → (B, 256,  56,  56)   [638]
  ├─ layer2                        → (B, 512,  28,  28)   [639]
  ├─ layer3                        → (B,1024,  14,  14)   [640]  = x_3   ★ điểm rẽ nhánh distill
  │
  │   ════════ NHÁNH DISTILL bắt đầu từ x_3 ════════
  ├─ reshape(x_3, (B, C, -1))      → (B,1024, 196)        [644]  duỗi lưới 14×14 = 196
  ├─ permute(0, 2, 1)              → (B, 196,1024)        [646]  = tmp  → mỗi ô lưới = 1 "token" 1024 chiều
  │
  ├─ pca_proj(tmp)   (khối Attention, xem chi tiết bên dưới)     [648]
  │     └─ trả về: _, attn_qk, attn_vv
  │        attn_vv lúc này = dots_vv → (B, 16, 196, 196)         (16 head, TRƯỚC softmax)
  │
  ├─ attn_vv.sum(dim=1) / num_heads → (B, 196, 196)      [651]  = A^{vv}_S  ★ trung bình 16 head
  │
  └─ đóng gói vào: return x, [attn_qk, attn_vv], vit_feat, ...   [666]
        → attn_weights[0] = A^{qk}_S ;  attn_weights[1] = A^{vv}_S
```

**`A^{qk}_S` và `A^{vv}_S` ra cùng 1 lần gọi `pca_proj(tmp)`** — chỉ khác nhau ở nhánh tính bên trong.

---

## Bên trong `pca_proj` — khối `Attention.forward` ([`resnet.py:165-181`](cakd_modified_files/resnet.py))

```
tmp (B, 196, 1024)
  │
  ├─ to_qkv(tmp)          Linear(1024 → 3072, bias=False)     [157]  sinh Q,K,V cùng lúc
  │     .chunk(3, dim=-1) → q, k, v, mỗi cái (B, 196, 1024)   [166]
  │
  ├─ rearrange 'b n (h d) -> b h n d', h=16                   [168]  tách 16 head
  │     → q, k, v mỗi cái (B, 16, 196, 64)     (d_h = 64)
  │
  ├─ dots_qk = matmul(q, kᵀ) * scale                         [171]  → A^{qk}_S (attention chuẩn)
  │
  ├─ dots_vv = matmul(v, vᵀ) * scale                         [173]  → A^{vv}_S  ★ CÁI TA QUAN TÂM
  │     → (B, 16, 196, 196)         ← "value nào giống value nào", CHƯA softmax
  │
  └─ return  self.to_out(out),  dots_qk,  dots_vv            [181]
                                          ↑↑↑
                              đây chính là attn_vv nhận ở dòng 648
```

---

## Giải thích KỸ (tập trung vào nhánh VV)

### 1–2. Sinh Q,K,V và tách 16 head — GIỐNG HỆT `A^{qk}_S`

```python
qkv = self.to_qkv(tmp).chunk(3, dim=-1)                          # q, k, v: (B,196,1024)  [166]
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=16), qkv)  # (B,16,196,64) [168]
```

- `to_qkv`: Linear(1024 → 3072) sinh Q, K, V cùng lúc, `.chunk(3)` cắt thành 3 phần (B,196,1024).
- `rearrange`: chia 1024 = 16 head × 64 chiều → q, k, v mỗi cái (B, 16, 196, 64).

Tới đây `v` đã sẵn sàng. **`A^{vv}_S` chỉ dùng `v`, không đụng tới `q`, `k`.**

### 3. `dots_vv = matmul(v, v.transpose) * scale` — trái tim của `A^{vv}_S`

```python
dots_vv = torch.matmul(v, v.transpose(-1, -2)) * self.scale   # [173]
```

- `v`: (B, 16, 196, **64**)
- `v.transpose(-1,-2)`: đổi 2 trục cuối → (B, 16, **64**, 196)
- `matmul`: (196, 64) × (64, 196) = **(196, 196)** cho mỗi head

```
matmul( (B,16,196,64) , (B,16,64,196) )  →  dots_vv (B, 16, 196, 196)
```

**Ma trận (196, 196) này là gì?** Phần tử hàng $i$, cột $j$ = **tích vô hướng** giữa Value của
ô $i$ và Value của ô $j$ = "**nội dung (value)** của 2 ô này giống/liên quan nhau tới đâu".
Khác với $A^{qk}_S$ (đo "ai **hỏi** hợp ai **trả lời**"), $A^{vv}_S$ đo "**nội dung** ai giống nội dung ai".

**`* self.scale`**: `scale = 64^(-0.5) = 1/8` — chia $\sqrt{d_h}$ giữ giá trị ở mức vừa phải
(y hệt lý do ở nhánh QK).

Kết quả `dots_vv`: **(B, 16, 196, 196)**, **CHƯA qua softmax** (điểm thô).

### 4. Trả về & trung bình 16 head

```python
_, attn_qk, attn_vv = self.pca_proj(tmp)      # attn_vv = dots_vv  [resnet.py:648]
attn_vv = attn_vv.sum(dim=1) / num_heads      # (B,16,196,196) → (B,196,196)  [651]
```

- Hàm trả về `dots_vv` ở vị trí thứ 3 → nhận vào biến `attn_vv`.
- `.sum(dim=1)/num_heads` cộng 16 head rồi chia 16 = trung bình → **(B, 196, 196)** = $A^{vv}_S$.

> **Chú ý (bất đối xứng softmax):** giống `A^{qk}_S`, `A^{vv}_S` là **điểm THÔ trước softmax**
> (softmax `self.attend` chỉ áp cho nhánh tính `out`, thứ bị vứt bằng `_`).

---

## Khác biệt DUY NHẤT so với `A^{qk}_S`

Mọi bước đều giống hệt, chỉ đổi **1 dòng công thức**:

| | `A^{qk}_S` (`attn_weights[0]`) | `A^{vv}_S` (`attn_weights[1]`) |
|---|---|---|
| Công thức | `matmul(q, kᵀ) * scale` [dòng 171] | `matmul(v, vᵀ) * scale` [dòng 173] |
| Dùng | Query × Key | Value × Value |
| Đo cái gì | "ô nào **chú ý** (query→key) ô nào" | "ô nào có **nội dung (value)** giống ô nào" |
| Trung bình head tại | dòng 650 | dòng 651 |
| Khớp với teacher | $\bar A^{(1)}_T$ = `tea_attn_weights[2]` | $\bar A^{(2)}_T$ = `tea_attn_weights[3]` |
| Trọng số trong `pca_loss` | **0.2** | **0.05** (nhẹ hơn) |

Công thức gọn:

$$
\texttt{dots\_vv}_h = \frac{V_S^{(h)} \big(V_S^{(h)}\big)^{\top}}{\sqrt{d_h}}
\qquad\Longrightarrow\qquad
\boxed{\; A^{vv}_S = \frac{1}{H}\sum_{h=1}^{H=16} \texttt{dots\_vv}_h \;}
$$

---

## Vai trò của `A^{vv}_S`

- Là **tín hiệu distill phụ**: bài Transformer thường KHÔNG có nhánh V·Vᵀ này — đây là **thêm thắt
  riêng của CAKD** để student bắt chước teacher kỹ hơn (khớp cả "quan hệ nội dung" giữa các patch).
- Dùng **1 chỗ duy nhất**: số hạng thứ 2 của `pca_loss` ([`dist_train_cakd.py:125-127`](dist_train_cakd.py)):

  ```python
  pca_loss = 0.2  * mse(attn_weights[0], tea_attn_weights[2][:,1:,1:].detach()) \
           + 0.05 * mse(attn_weights[1], tea_attn_weights[3][:,1:,1:].detach())
                    #        ↑↑↑ A^{vv}_S khớp với attention lớp cuối (phần 2) của teacher
  ```

- **KHÔNG** đưa vào discriminator (GAN) — chỉ `A^{qk}_S` mới làm "mẫu giả" cho GAN.

---

*Nguồn: `ResNet_CAKD._forward_impl` (dòng 631–666) và `Attention.forward` (dòng 165–181)
trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py); `pca_loss` tại
[`dist_train_cakd.py:123-127`](dist_train_cakd.py).*
