# `f_S` (patch feature của student) — từ ảnh gốc ra feature qua những bước nào?

> Giải thích biến `proj_feat` (vit_feat) trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $f_S$ | `proj_feat` (vit_feat) | (B, 196, 768) | [`resnet.py:652`](cakd_modified_files/resnet.py) `vit_feat = self.gl_proj(tmp)` | đầu ra `gl_proj` (GLProj) |

`f_S` là **feature theo từng patch** của student, shape (B, 196, 768): **196 token, mỗi token
768 chiều** — cố ý làm cho **giống định dạng feature của teacher ViT** (`tea_feat` cũng (B, 196, 768)).
Nó ra từ **nhánh distill**, qua một lớp "phiên dịch" đặc biệt tên `gl_proj` (kiểu **GLProj =
Group-wise Linear projection**), biến đặc trưng CNN (1024 chiều) → không gian ViT (768 chiều).

---

## Hành trình `image → f_S`

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
  ├─ gl_proj(tmp)    (khối GLProj, xem chi tiết bên dưới)        [652]
  │     → vit_feat  (B, 196, 768)      ← "phiên dịch" 1024 → 768 (chiều ViT)
  │
  └─ đóng gói vào: return x, [...], vit_feat, ...               [666]
        → proj_feat = f_S
```

> Lưu ý: `f_S` dùng **CÙNG đầu vào `tmp`** với `A^{qk}_S` / `A^{vv}_S` (đều là chuỗi 196 token 1024 chiều),
> chỉ khác là đưa qua `gl_proj` thay vì `pca_proj`.

---

## Bên trong `gl_proj` — khối `GLProj` ([`resnet.py:183-222`](cakd_modified_files/resnet.py))

`gl_proj = GLProj(src_dim=1024, tgt_dim=768, num_patch=196)` ([`resnet.py:566`](cakd_modified_files/resnet.py)).

**Ý tưởng cốt lõi:** thay vì dùng **1** lớp Linear chung cho cả 196 patch, GLProj **chia 196 patch
thành 16 NHÓM** (theo vị trí trên lưới), **mỗi nhóm có 1 lớp Linear RIÊNG** (1024 → 768).
Nhờ vậy mỗi vùng ảnh được "phiên dịch" theo cách riêng → khớp ViT mềm dẻo hơn.

### Lúc khởi tạo (`__init__`, [`resnet.py:189-202`](cakd_modified_files/resnet.py))

```python
if num_patch == 196:  num_fc = 16      # 196 patch → 16 nhóm → 16 lớp Linear
for i in range(num_fc):
    layers[f"fc_layer_{i}"] = nn.Linear(1024, 768)   # mỗi nhóm 1 Linear riêng
```

Tạo ra **16 lớp Linear**, mỗi cái `Linear(1024, 768)`, tên `fc_layer_0 … fc_layer_15`.

### Lúc chạy (`forward`, [`resnet.py:204-222`](cakd_modified_files/resnet.py))

```python
def forward(self, x):                                  # x = tmp (B, 196, 1024)
    out = torch.zeros((B, 196, 768)).to('cuda')        # [207] tạo khung rỗng để ghi kết quả
    idx = idx_196                                       # [211] bảng 16 nhóm chỉ số patch
    for i in range(16):
        out[:, idx[i], :] = self.layers[i](x[:, idx[i], :])   # [221] mỗi nhóm qua Linear riêng
    return out                                          # (B, 196, 768) = vit_feat
```

Diễn giải vòng lặp (làm 16 lần, mỗi lần cho 1 nhóm `i`):

```
x[:, idx[i], :]   = lấy đúng các patch thuộc nhóm i     (B, |nhóm i|, 1024)
     │  self.layers[i]  (= fc_layer_i: Linear 1024→768)
     ▼
                                                        (B, |nhóm i|, 768)
     │  ghi vào đúng vị trí cũ trong out
     ▼
out[:, idx[i], :] = kết quả
```

Sau 16 vòng, mọi patch đều đã được điền → `out` (B, 196, 768) = `vit_feat` = $f_S$.

---

## `idx_196` là gì? — bảng chia 196 patch thành 16 nhóm

`idx_196` ([`resnet.py:85-101`](cakd_modified_files/resnet.py)) là **list gồm 16 list con**, mỗi list con
liệt kê **chỉ số các patch (0..195) cùng thuộc 1 nhóm**. Ví dụ:

```python
idx_224_16_0 = [0,1,2,3, 14,15,16,17, 28,29,30,31, 42,43,44,45]   # nhóm 0 (16 patch)
idx_224_16_1 = [4,5,6,7, 18,19,20,21, 32,33,34,35, 46,47,48,49]   # nhóm 1
...
idx_224_16_15 = [180,181,194,195]                                 # nhóm 15 (4 patch)
idx_196 = [idx_224_16_0, idx_224_16_1, ..., idx_224_16_15]        # gộp 16 nhóm
```

- Các con số này là **bảng tra cứng** (tính sẵn) — gom những ô **gần nhau trên lưới 14×14** vào cùng nhóm
  (mỗi nhóm ≈ 1 vùng vuông của ảnh). Tổng số patch của 16 nhóm = 196.
- Kích thước nhóm **không đều nhau** (nhóm 0 có 16 patch, nhóm 15 chỉ 4) — do cách chia lưới, nhưng
  không sao: mỗi nhóm có Linear riêng nên xử lý được số patch bất kỳ.

> Ghi chú thực thi: dòng 207 `torch.zeros(...).to('cuda')` **hardcode GPU** → `gl_proj` **bắt buộc
> chạy trên GPU** (chạy CPU sẽ lỗi lệch device).

---

## Công thức gọn

Ký hiệu nhóm $g \in \{0,\dots,15\}$, $\mathcal{I}_g$ = tập chỉ số patch của nhóm $g$ (`idx_196[g]`),
$W_g, b_g$ = trọng số của `fc_layer_g`. Với mỗi patch $p \in \mathcal{I}_g$:

$$
\boxed{\; f_S[:,\,p,\,:] \;=\; W_g \cdot \texttt{tmp}[:,\,p,\,:] \;+\; b_g \;}
\qquad (1024 \to 768)
$$

Mỗi patch được chiếu bằng **Linear của nhóm chứa nó** — đó là ý nghĩa "group-wise".

---

## Vai trò của `f_S`

Dùng **1 chỗ duy nhất**: số hạng thứ 3 của `gl_loss` ([`dist_train_cakd.py:132`](dist_train_cakd.py)):

```python
gl_loss = mse(output, tea_logits.detach())      \
        + mse(proj_token, tea_token)            \
        + 0.05 * mse(proj_feat, tea_feat.detach())   # ← f_S khớp với patch feature teacher (nhẹ, 0.05)
```

- **Ép `f_S` (patch feature student) giống `tea_feat` (patch feature teacher ViT)** bằng MSE.
- Cả hai đều (B, 196, 768) → khớp 1-1 từng patch, từng chiều.
- Hệ số **0.05** (nhẹ) — feature theo patch chỉ là tín hiệu phụ, không nặng bằng logits/token (hệ số 1).
- `tea_feat.detach()` → không truyền gradient về teacher.
- **KHÔNG** đưa vào GAN.

> So sánh: `gl_proj` sinh **feature theo patch** (196 token), còn `cls_proj` sinh **1 token tổng thể**
> (`t_S`, chiều 768) để khớp class token teacher. Hai cái bổ trợ nhau trong `gl_loss`.

---

## Bảng số chiều (cho `gl_proj`)

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| `src_dim` (chiều vào) | 1024 (= kênh ra `layer3`) | `self.inplanes` sau layer3 |
| `tgt_dim` (chiều ra) | 768 (= chiều ViT-B/16) | [`resnet.py:566`](cakd_modified_files/resnet.py) |
| `num_patch` | 196 | [`resnet.py:566`](cakd_modified_files/resnet.py) |
| số nhóm / số Linear (`num_fc`) | 16 | [`resnet.py:195`](cakd_modified_files/resnet.py) |
| mỗi Linear | `Linear(1024, 768)` | [`resnet.py:201`](cakd_modified_files/resnet.py) |
| bảng chia nhóm | `idx_196` (16 nhóm) | [`resnet.py:85-101`](cakd_modified_files/resnet.py) |

*(Nếu `num_patch=49` thì dùng 4 nhóm với bảng `idx_49`; nếu số khác thì dùng **1** Linear chung —
xem [`resnet.py:194-217`](cakd_modified_files/resnet.py).)*

---

*Nguồn: `ResNet_CAKD._forward_impl` (dòng 631–666) và `GLProj` (dòng 183–222)
trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py); `gl_loss` tại
[`dist_train_cakd.py:129-133`](dist_train_cakd.py).*
