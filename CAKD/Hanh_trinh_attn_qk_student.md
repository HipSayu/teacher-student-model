# `A^{qk}_S` (attention QK của student) — từ ảnh gốc ra attention map qua những bước nào?

> Giải thích biến `attn_weights[0]` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $A^{qk}_S$ | `attn_weights[0]` | (B, 196, 196) | [`resnet.py:648-650`](cakd_modified_files/resnet.py) | $\frac{1}{H}\sum_h \frac{Q_S K_S^\top}{\sqrt{d_h}}$ — **TRƯỚC softmax**, đã trung bình 16 head |

`A^{qk}_S` là **ma trận attention (B, 196, 196)** cho biết "ô lưới nào chú ý tới ô lưới nào".
Nó **KHÔNG** ra từ nhánh phân loại (`fc`), mà ra từ **nhánh distill** — nơi CNN được "độ thêm"
một khối Attention kiểu Transformer (`pca_proj`) để **bắt chước attention của teacher ViT**.
Dưới đây là toàn bộ hành trình từ ảnh gốc, kèm shape từng bước.

---

## Hành trình `image → A^{qk}_S`

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
  │                                                              (đúng định dạng ViT: (batch, số_token, chiều))
  │
  ├─ pca_proj(tmp)   (khối Attention, xem chi tiết bên dưới)     [648]
  │     └─ trả về: _, attn_qk, attn_vv
  │        attn_qk lúc này = dots_qk  → (B, 16, 196, 196)        (16 head, TRƯỚC softmax)
  │
  ├─ attn_qk.sum(dim=1) / num_heads → (B, 196, 196)      [650]  = A^{qk}_S  ★ trung bình 16 head
  │
  └─ đóng gói vào: return x, [attn_qk, attn_vv], vit_feat, ...   [666]
        → attn_weights[0] = A^{qk}_S ;  attn_weights[1] = A^{vv}_S
```

---

## Bên trong `pca_proj` — khối `Attention.forward` ([`resnet.py:165-181`](cakd_modified_files/resnet.py))

`pca_proj = Attention(dim=1024, heads=16, dim_head=64)` ([`resnet.py:564`](cakd_modified_files/resnet.py)).
Đầu vào là `tmp` (B, 196, 1024).

```
tmp (B, 196, 1024)
  │
  ├─ to_qkv(tmp)          Linear(1024 → 3072, bias=False)     [157]  sinh Q,K,V cùng lúc
  │     .chunk(3, dim=-1) → q, k, v, mỗi cái (B, 196, 1024)   [166]
  │
  ├─ rearrange 'b n (h d) -> b h n d', h=16                   [168]  tách 16 head
  │     → q, k, v mỗi cái (B, 16, 196, 64)     (d_h = 64)
  │
  ├─ dots_qk = matmul(q, k.transposeᵀ) * scale               [171]  scale = 64^(-0.5) = 1/8
  │     → (B, 16, 196, 196)         ← ĐIỂM TƯƠNG QUAN THÔ (query·key), CHƯA softmax
  │
  ├─ (song song) dots_vv = matmul(v, vᵀ) * scale             [173]  → A^{vv}_S (tín hiệu distill thứ 2)
  │
  └─ return  self.to_out(out),  dots_qk,  dots_vv            [181]
                                  ↑↑↑
                     đây chính là attn_qk nhận ở dòng 648
```

> **Điểm mấu chốt:** hàm trả về `dots_qk` — tức **điểm thô $Q K^\top/\sqrt{d_h}$ TRƯỚC softmax**.
> Phép `self.attend` (softmax, dòng 175) **chỉ** áp cho nhánh tính `out` (dùng để tổng hợp value),
> **không** áp cho `dots_qk` trả về. Vì vậy `A^{qk}_S` là **logit attention**, không phải xác suất.

---

## Giải thích KỸ từng dòng (đọc nếu chưa hiểu sơ đồ trên)

### 0. Ý tưởng: Q, K, V là gì? (đọc cái này trước)

Tưởng tượng 196 ô lưới như 196 người trong phòng họp. Mỗi người muốn thu thập thông tin từ
những người liên quan. Attention cho mỗi người **3 vai trò**, mỗi vai trò là 1 vector:

- **Q (Query)** = "tôi đang **tìm** gì?" — câu hỏi của tôi.
- **K (Key)** = "tôi **chứa** thông tin gì?" — nhãn dán để người khác tra.
- **V (Value)** = "nội dung **thực sự** của tôi" — thứ sẽ được truyền đi.

Cách hoạt động: người $i$ lấy **Query của mình** so với **Key của mọi người** → ra điểm
"tôi hợp với ai". Ai điểm cao thì tôi lấy nhiều **Value** của người đó. Đoạn code này chính là
biến ý tưởng đó thành phép nhân ma trận.

### 1. `to_qkv(tmp)` — sinh Q, K, V cùng một lúc

```python
self.to_qkv = nn.Linear(1024, 3072, bias=False)   # [resnet.py:157]
qkv = self.to_qkv(tmp).chunk(3, dim=-1)            # [resnet.py:166]
```

- `tmp` là (B, 196, **1024**) — mỗi ô là 1 vector 1024 chiều.
- `to_qkv` là 1 lớp Linear biến 1024 → **3072**. Vì sao 3072? Vì `3072 = 1024 × 3` — ta cần
  **3 thứ** (Q, K, V), mỗi thứ 1024 chiều. Làm 1 phép Linear to rồi cắt ra cho nhanh, thay vì 3 phép riêng.
- Kết quả: (B, 196, 3072).

`.chunk(3, dim=-1)` = **cắt trục cuối (3072) thành 3 khúc bằng nhau**, mỗi khúc 1024:

```
(B, 196, 3072)  ──chunk(3)──►   q: (B, 196, 1024)
                                k: (B, 196, 1024)
                                v: (B, 196, 1024)
```

Giờ mỗi ô đã có đủ 3 vai trò Q, K, V (mỗi vai trò 1024 chiều).

### 2. `rearrange 'b n (h d) -> b h n d', h=16` — tách 16 "đầu" (multi-head)

```python
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=16), qkv)  # [168]
```

Đây là chỗ dễ rối nhất. Ý tưởng: thay vì 1 attention nhìn cả 1024 chiều, ta **chia 1024 thành
16 nhóm nhỏ 64 chiều** (16 × 64 = 1024), mỗi nhóm là 1 "đầu" (head) attention **độc lập**,
nhìn dữ liệu theo 1 góc khác nhau. Nhiều góc nhìn → phong phú hơn.

Cú pháp `einops`:
- `b n (h d)` = shape đầu vào **(B, 196, 1024)**, trong đó `1024` được **hiểu ngầm là h×d = 16×64**.
- `-> b h n d` = sắp lại thành **(B, 16, 196, 64)**.

Ý nghĩa các chữ:

| Chữ | Là gì | Giá trị |
|---|---|---|
| `b` | batch | B |
| `n` | số token (ô lưới) | 196 |
| `h` | số head | 16 |
| `d` | chiều mỗi head ($d_h$) | 64 |

```
q: (B, 196, 1024)  ──tách──►  (B, 16, 196, 64)
                              └─ 16 head, mỗi head lo 196 token × 64 chiều
```

Làm y hệt cho k và v (`map` = áp cùng phép cho cả 3). Xong bước này: `q, k, v` đều là **(B, 16, 196, 64)**.

### 3. `dots_qk = matmul(q, k.transpose) * scale` — tính điểm "ai chú ý ai"

```python
dots_qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [171]
```

Với **mỗi head**, ta nhân ma trận `q` với `k` đã **chuyển vị** (transpose 2 trục cuối):

- `q`: (B, 16, 196, **64**)
- `k.transpose(-1,-2)`: đổi 2 trục cuối của k → (B, 16, **64**, 196)
- `matmul`: (196, 64) × (64, 196) = **(196, 196)**

```
matmul( (B,16,196,64) , (B,16,64,196) )  →  (B, 16, 196, 196)
```

**Ma trận (196, 196) này là gì?** Phần tử ở hàng $i$, cột $j$ = **tích vô hướng** giữa Query của
ô $i$ và Key của ô $j$ = "ô $i$ hợp/liên quan với ô $j$ tới mức nào". Càng lớn = càng chú ý.
Đây chính là công thức $Q K^\top$ — mỗi token so với **tất cả** token khác, nên ra ma trận vuông 196×196.

**`* self.scale` là gì?** `scale = 64^(-0.5) = 1/8` ([`resnet.py:151`](cakd_modified_files/resnet.py)).
Chia cho $\sqrt{d_h}=\sqrt{64}=8$. Lý do: khi cộng 64 số lại, tích vô hướng có thể ra rất lớn;
chia $\sqrt{64}$ để **kéo giá trị về mức vừa phải**, tránh softmax sau này bị "bão hòa"
(dồn hết vào 1 ô). Đây là chuẩn của Transformer gốc.

Kết quả `dots_qk`: **(B, 16, 196, 196)** — 16 ma trận attention (mỗi head 1 cái),
**CHƯA qua softmax** (vẫn là điểm thô, có thể âm/dương).

### 4. `dots_vv = matmul(v, v.transpose) * scale` — tín hiệu distill thứ 2

```python
dots_vv = torch.matmul(v, v.transpose(-1, -2)) * self.scale   # [173]
```

Giống hệt bước 3 nhưng dùng **V với chính V** (thay vì Q với K). Ra ma trận (B, 16, 196, 196)
đo "value của các ô giống nhau tới đâu". Đây là **biến thể riêng của CAKD** — dùng làm tín hiệu
distill thứ hai ($A^{vv}_S$), để khớp thêm với teacher. Bài Transformer thường **không** có cái này.

### 5. `return self.to_out(out), dots_qk, dots_vv` — trả về 3 thứ

```python
attn_qk = self.attend(dots_qk)   # [175] softmax → trọng số THẬT
attn = self.dropout(attn_qk)
out = torch.matmul(attn, v)      # [178] dùng trọng số tổng hợp value
out = rearrange(out, 'b h n d -> b n (h d)')  # [179] ghép 16 head lại
return self.to_out(out), dots_qk, dots_vv     # [181]
```

Hàm trả về **3 thứ**, nhưng nhánh distill chỉ lấy 2 cái sau:

```python
_, attn_qk, attn_vv = self.pca_proj(tmp)   # [resnet.py:648]
```

- **Thứ 1** `self.to_out(out)` → bị vứt (dấu `_`). Đây là output attention "chuẩn"
  (đã softmax + tổng hợp value). Nhánh phân loại của CAKD không cần nó.
- **Thứ 2** `dots_qk` → **nhận vào biến `attn_qk`** ở dòng 648. Đây là thứ ta quan tâm.
- **Thứ 3** `dots_vv` → nhận vào `attn_vv`.

> **Điểm CỰC KỲ quan trọng, hay nhầm:** biến `attn_qk` mà nhánh distill nhận về **chính là
> `dots_qk` — điểm THÔ, CHƯA softmax**. Softmax (`self.attend`, dòng 175) chỉ áp cho `attn` để
> tính `out` (thứ bị vứt), **không** áp cho `dots_qk` trả về. Vì thế attention của student là
> "logit", còn teacher lại là xác suất (đã softmax) → đây là chỗ "bất đối xứng softmax".

### 6. Sau khi rời hàm: trung bình 16 head

Về lại `_forward_impl` ([`resnet.py:650`](cakd_modified_files/resnet.py)):

```python
attn_qk = attn_qk.sum(dim=1) / num_heads   # (B,16,196,196) → (B,196,196)
```

`dim=1` là trục head. Cộng 16 head lại rồi chia 16 = **lấy trung bình 16 head** → gộp thành
**1 ma trận attention (B, 196, 196)** duy nhất = $A^{qk}_S$. Xong.

### Tóm tắt 1 dòng mỗi bước

| Bước | Vào | Ra | Làm gì |
|---|---|---|---|
| `to_qkv` + `chunk` | (B,196,1024) | q,k,v: (B,196,1024) | sinh 3 vai trò Q,K,V |
| `rearrange` | (B,196,1024) | (B,16,196,64) | chia 16 head |
| `matmul(q,kᵀ)*scale` | q,k | dots_qk (B,16,196,196) | điểm "ai chú ý ai", chưa softmax |
| `matmul(v,vᵀ)*scale` | v | dots_vv (B,16,196,196) | tín hiệu distill phụ |
| `return` | — | dots_qk, dots_vv | trả điểm thô cho distill |
| `sum/num_heads` | (B,16,196,196) | (B,196,196) | trung bình head → $A^{qk}_S$ |

---

## Công thức gọn

$$
Q_S, K_S, V_S = \texttt{to\_qkv(tmp).chunk(3)}, \qquad
\texttt{scale} = d_h^{-1/2} = 64^{-1/2}
$$

$$
\texttt{dots\_qk}_h = \frac{Q_S^{(h)} \big(K_S^{(h)}\big)^{\top}}{\sqrt{d_h}}
\quad\text{(mỗi head }h,\ \text{shape (B,196,196))}
$$

$$
\boxed{\; A^{qk}_S = \frac{1}{H}\sum_{h=1}^{H=16} \texttt{dots\_qk}_h \;}
\qquad \text{(dòng 650: } \texttt{attn\_qk.sum(dim=1)/num\_heads} \text{)}
$$

---

## Ý nghĩa & vai trò

- **196 = 14×14**: lưới đặc trưng của `layer3` được coi như 196 "patch/token", đúng bằng số patch
  của teacher ViT (ảnh 224 chia lưới 14×14). Nhờ vậy attention student (196×196) **khớp 1-1**
  với attention teacher để so bằng MSE.
- **Ma trận (196, 196)**: phần tử $[i, j]$ = mức độ token $i$ (query) chú ý tới token $j$ (key).
- **`A^{qk}_S` được dùng ở 2 chỗ:**
  1. `pca_loss` — ép giống attention teacher $\bar A^{(1)}_T$ ([`dist_train_cakd.py:123`](dist_train_cakd.py)).
  2. Đưa vào **discriminator** (GAN) làm "mẫu giả" `input_d_fake` ([`dist_train_cakd.py:110`](dist_train_cakd.py)),
     student cố ép $D$ tưởng nó là "thật".

> ⚠️ **Bất đối xứng softmax:** `A^{qk}_S` là điểm **trước softmax**, nhưng teacher $\bar A^{(1)}_T$
> là xác suất **sau softmax**. `pca_loss` (MSE) ép hai thứ khác thang đo — xem mục 6, điểm 7 trong
> [`FORMULAS.md`](FORMULAS.md).

---

## Bảng số chiều (cho `pca_proj`)

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| `dim` (chiều token vào) | 1024 (= kênh ra `layer3`) | `self.inplanes` sau layer3 |
| `heads` (H) | 16 | [`resnet.py:564`](cakd_modified_files/resnet.py) |
| `dim_head` ($d_h$) | 64 (= 1024/16) | [`resnet.py:564`](cakd_modified_files/resnet.py) |
| `inner_dim` | 1024 (= 64×16) | [`resnet.py:147`](cakd_modified_files/resnet.py) |
| `scale` | 1/8 (= 64^(-0.5)) | [`resnet.py:151`](cakd_modified_files/resnet.py) |
| `to_qkv` | Linear(1024 → 3072) | [`resnet.py:157`](cakd_modified_files/resnet.py) |

---

*Nguồn: `ResNet_CAKD._forward_impl` (dòng 631–666) và `Attention.forward` (dòng 165–181)
trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py).*
