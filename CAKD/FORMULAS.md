# CAKD — Công thức loss CHUẨN theo code

> Đối chiếu với bài báo *Cross-Architecture Knowledge Distillation* (arXiv:2207.05273).
> Các công thức dưới đây được viết lại cho **đúng 100% với code thực tế** trong
> [`dist_train_cakd.py`](dist_train_cakd.py) (loss),
> [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py) (student) và
> [`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py) (teacher).
> Mọi ký hiệu đều **bám sát tên biến trong code** và ghi rõ **file:dòng** nơi nó sinh ra.
> Những chỗ bài báo viết sai/thiếu được liệt kê ở cuối.

---

## 0. Bức tranh tổng thể (1 batch)

```
                 image (B, 3, 224, 224)
                    │
        ┌───────────┴─────────────┐
        ▼                          ▼
  STUDENT  ResNet_CAKD        TEACHER  ViT-B/16  (eval, đóng băng)
  model(image)                teacher(image)
        │                          │
  output, attn_weights,      tea_logits, tea_attn_weights,
  proj_feat, proj_token      tea_token, tea_feat
        │                          │
        └────────── 4 loss ────────┘
   cls_loss · pca_loss · gl_loss · gan_loss
        │
   2 optimizer xen kẽ:  d_optimizer (D)  →  optimizer (student)
```

Đầu ra student — `output, attn_weights, proj_feat, proj_token = model(image)` — ở
[`dist_train_cakd.py:98`](dist_train_cakd.py).
Đầu ra teacher — `tea_logits, tea_attn_weights, tea_token, tea_feat = teacher(image)` — ở
[`dist_train_cakd.py:100`](dist_train_cakd.py).

Ký hiệu chung: **B** = batch size, **N** = 196 patch (= lưới 14×14), **C** = số kênh
đầu ra `layer3` của ResNet-50 (= 1024), **d** = 768 (chiều đặc trưng ViT-B/16).

---

## 1. Ký hiệu (bám sát biến trong code)

### 1.1 Đầu ra STUDENT — `model(image)` → 4 thứ

| Ký hiệu | Biến trong code | Shape | Sinh ra tại | Ghi chú |
|---|---|---|---|---|
| $z_S$ | `output` | (B, 1000) | [`resnet.py:659`](cakd_modified_files/resnet.py) `x = self.fc(cnn_token)` | logits phân loại |
| $A^{qk}_S$ | `attn_weights[0]` | (B, 196, 196) | [`resnet.py:648-650`](cakd_modified_files/resnet.py) | $\frac{1}{H}\sum_h \frac{Q_S K_S^\top}{\sqrt{d_h}}$ — **TRƯỚC softmax**, đã trung bình 16 head |
| $A^{vv}_S$ | `attn_weights[1]` | (B, 196, 196) | [`resnet.py:648,651`](cakd_modified_files/resnet.py) | $\frac{1}{H}\sum_h \frac{V_S V_S^\top}{\sqrt{d_h}}$ — **TRƯỚC softmax** |
| $f_S$ | `proj_feat` (vit_feat) | (B, 196, 768) | [`resnet.py:652`](cakd_modified_files/resnet.py) `vit_feat = self.gl_proj(tmp)` | đầu ra `gl_proj` (GLProj) |
| $t_S$ | `proj_token` (cls_token) | (B, 768) | [`resnet.py:666`](cakd_modified_files/resnet.py) `self.cls_proj(cnn_token)` | đầu ra `cls_proj` (Linear) |

$H = 16$ head (`self.pca_proj = Attention(dim=C, heads=16, dim_head=C/16)`,
[`resnet.py:564`](cakd_modified_files/resnet.py)); $d_h = C/16$ ⇒ `scale` $= d_h^{-1/2}$
([`resnet.py:151`](cakd_modified_files/resnet.py)).

### 1.2 Đầu ra TEACHER — `teacher(image)` → 4 thứ

| Ký hiệu | Biến trong code | Shape | Sinh ra tại | Ghi chú |
|---|---|---|---|---|
| $z_T$ | `tea_logits` | (B, 1000) | [`vision_transformer.py:367`](cakd_modified_files/vision_transformer.py) | logits teacher, `.detach()` khi dùng |
| $\bar A^{(1)}_T$ | `tea_attn_weights[2][:, 1:, 1:]` | (B, 196, 196) | [`vision_transformer.py:209`](cakd_modified_files/vision_transformer.py) | attention **lớp cuối**, **SAU softmax**, bỏ CLS |
| $\bar A^{(2)}_T$ | `tea_attn_weights[3][:, 1:, 1:]` | (B, 196, 196) | [`vision_transformer.py:209`](cakd_modified_files/vision_transformer.py) | attention **lớp cuối** (phần 2), bỏ CLS |
| $f_T$ | `tea_feat` | (B, 196, 768) | [`vision_transformer.py:365`](cakd_modified_files/vision_transformer.py) `feats = x[:, 1:]` | 196 patch token, `.detach()` khi dùng |
| $t_T$ | `tea_token` | (B, 768) | [`vision_transformer.py:364`](cakd_modified_files/vision_transformer.py) `cls_token = x[:, 0]` | class token, **KHÔNG** detach |

`tea_attn_weights` là **list 4 phần tử** trả về ở [`vision_transformer.py:209`](cakd_modified_files/vision_transformer.py):
`[attn_weights_2[0], attn_weights_2[1], attn_weights_1[0], attn_weights_1[1]]`
— index `[0][1]` = lớp **áp chót**, index `[2][3]` = lớp **cuối**. Code chỉ dùng `[2]` và `[3]`
(hai attention của **lớp cuối**). `[:, 1:, 1:]` = bỏ hàng/cột của class token (197→196).

### 1.3 Discriminator (GAN)

| Ký hiệu | Biến trong code | Shape | Ghi chú |
|---|---|---|---|
| `input_d_real` | $\bar A^{(1)}_T$ + `[:,None,:,:]` + `.detach()` | (B, 1, 196, 196) | "mẫu thật" — [`dist_train_cakd.py:106`](dist_train_cakd.py) |
| `input_d_fake` | $A^{qk}_S$ + `[:,None,:,:]` + `.detach()` | (B, 1, 196, 196) | "mẫu giả" — [`dist_train_cakd.py:110`](dist_train_cakd.py) |
| `pred_real` | `discriminator(input_d_real)` | (B, 1, h, w) | điểm cho attention teacher — [`dist_train_cakd.py:113`](dist_train_cakd.py) |
| `pred_fake` | `discriminator(input_d_fake)` | (B, 1, h, w) | điểm cho attention student — [`dist_train_cakd.py:114`](dist_train_cakd.py) |
| $D(\cdot)$ | `discriminator` | — | `NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3)` (PatchGAN), [`dist_train_cakd.py:419`](dist_train_cakd.py) |

### 1.4 Quy ước hàm loss

- $\mathrm{MSE}(a,b) = \frac{1}{n}\lVert a-b\rVert_2^2$ — `mse_criterion = nn.MSELoss()` ([`dist_train_cakd.py:433`](dist_train_cakd.py)).
- $\mathrm{CE}(z, y)$ — `criterion = nn.CrossEntropyLoss(label_smoothing=...)` ([`dist_train_cakd.py:430`](dist_train_cakd.py)).
- $\ell(p, c) = \mathrm{BCEWithLogits}\big(p,\; c\cdot\mathbf{1}\big)$, $c=1$ (real) hoặc $c=0$ (fake).
  `gan_criterion = GANLoss()` mặc định `gan_mode='vanilla'` → **BCEWithLogitsLoss**
  ([`new_utils.py:110`](new_utils.py)), KHÔNG phải LSGAN. Trong code viết là
  `gan_criterion(pred, True/False)`; nhãn `True`→1, `False`→0 do `GANLoss.get_target_tensor`.

---

## 2. Luồng sinh mỗi ký hiệu (data-flow bám biến)

### 2.1 Student: từ ảnh → 4 đầu ra (`ResNet_CAKD._forward_impl`, [`resnet.py:631-666`](cakd_modified_files/resnet.py))

```
image ─stem+layer1+layer2─► x ─layer3─► x_3            (B, C=1024, 14, 14)   [dòng 640]
                                          │
   ┌──────────────────────────────────────┴───────── NHÁNH DISTILL ─────────┐
   │  tmp = reshape(x_3) → (B, C, 196) → permute → (B, 196, C)   [641-646]   │
   │  _, attn_qk, attn_vv = self.pca_proj(tmp)                    [648]      │
   │      attn_qk = attn_qk.sum(dim=1)/num_heads  →  A^{qk}_S     [650]      │
   │      attn_vv = attn_vv.sum(dim=1)/num_heads  →  A^{vv}_S     [651]      │
   │  vit_feat  = self.gl_proj(tmp)               →  f_S          [652]      │
   └────────────────────────────────────────────────────────────────────────┘
   │  NHÁNH PHÂN LOẠI (dùng LẠI x_3, không phải tmp)                          │
   │  x = layer4(x_3) → avgpool → flatten = cnn_token  (B, 2048) [655-658]   │
   │  output   = self.fc(cnn_token)               →  z_S          [659]      │
   │  proj_token = self.cls_proj(cnn_token)       →  t_S          [666]      │
   └──return: output, [attn_qk, attn_vv], vit_feat, proj_token────────────────
```

Bên trong `pca_proj` (`Attention.forward`, [`resnet.py:165-181`](cakd_modified_files/resnet.py)):

$$
\underbrace{Q_S,K_S,V_S}_{\texttt{to\_qkv(x).chunk(3)}}
\;\Rightarrow\;
\texttt{dots\_qk}=\frac{Q_S K_S^\top}{\sqrt{d_h}},\quad
\texttt{dots\_vv}=\frac{V_S V_S^\top}{\sqrt{d_h}}
\qquad(\text{đều nhân }\texttt{self.scale}=d_h^{-1/2})
$$

`dots_qk`/`dots_vv` có shape (B, 16, 196, 196); sau khi `.sum(dim=1)/num_heads`
([`resnet.py:650-651`](cakd_modified_files/resnet.py)) thành $A^{qk}_S,A^{vv}_S$ shape (B, 196, 196).
**Quan trọng:** hàm trả về `dots_qk, dots_vv` là **điểm THÔ trước `self.attend` (softmax)** —
softmax chỉ áp cho nhánh tính `out`, không áp cho tín hiệu distill.

### 2.2 Teacher: từ ảnh → 4 đầu ra (`VisionTransformer.forward`, [`vision_transformer.py:351-374`](cakd_modified_files/vision_transformer.py))

```
image → _process_input → (B, 196, 768)                              [353]
      → cat[class_token, .] → (B, 197, 768)                          [358]
      → encoder(.) → x (B,197,768),  attn_weights (list 4)           [361]
      cls_token = x[:, 0]   → t_T   (B, 768)                          [364]
      feats     = x[:, 1:]  → f_T   (B, 196, 768)                     [365]
      logits    = heads(cls_token) → z_T  (B, 1000)                   [367]
```

`attn_weights` sinh trong `Encoder.forward` ([`vision_transformer.py:199-209`](cakd_modified_files/vision_transformer.py)):
2 lớp cuối bật `need_weights=True`, mỗi lớp cho ra ma trận attention **đã qua softmax**
(shape (B, 197, 197)). Code lấy `[2]`, `[3]` (lớp cuối) rồi cắt `[:, 1:, 1:]` → (B, 196, 196)
= $\bar A^{(1)}_T,\bar A^{(2)}_T$.

> ⚠️ **Bất đối xứng then chốt:** $A^{qk}_S$ là điểm **trước softmax**, còn $\bar A^{(1)}_T$ là xác suất
> **sau softmax**. `pca_loss` ép hai thứ khác thang đo bằng MSE (xem §4.7).

---

## 3. Bốn loss thành phần (đúng dòng code)

### (1) `cls_loss` — Phân loại — [`dist_train_cakd.py:120`](dist_train_cakd.py)

```python
cls_loss = criterion(output, target)
```

$$\boxed{\;\mathcal{L}_{cls} = \mathrm{CE}(z_S,\, y)\;}\qquad (\text{có label smoothing})$$

### (2) `pca_loss` — Khớp attention — [`dist_train_cakd.py:123-127`](dist_train_cakd.py)

```python
pca_loss = 0.2  * mse_criterion(attn_weights[0], tea_attn_weights[2][:, 1:, 1:].detach()) \
         + 0.05 * mse_criterion(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())
```

$$\boxed{\;\mathcal{L}_{pca} = 0.2\cdot\mathrm{MSE}\!\big(A^{qk}_S,\ \bar A^{(1)}_T\big)
\;+\; 0.05\cdot\mathrm{MSE}\!\big(A^{vv}_S,\ \bar A^{(2)}_T\big)\;}$$

Phía teacher `.detach()` (không truyền gradient về teacher).

### (3) `gl_loss` — Khớp logits + token + feature — [`dist_train_cakd.py:129-133`](dist_train_cakd.py)

```python
gl_loss = mse_criterion(output,     tea_logits.detach()) \
        + mse_criterion(proj_token, tea_token) \
        + 0.05 * mse_criterion(proj_feat, tea_feat.detach())
```

$$\boxed{\;\mathcal{L}_{gl} = \underbrace{\mathrm{MSE}(z_S,\, z_T)}_{\text{logits, hệ số }1}
\;+\; \underbrace{\mathrm{MSE}(t_S,\, t_T)}_{\text{class token, hệ số }1}
\;+\; \underbrace{0.05\cdot\mathrm{MSE}(f_S,\, f_T)}_{\text{patch feature, hệ số }0.05}\;}$$

Chi tiết `.detach()`: `tea_logits` và `tea_feat` **có** detach; riêng `tea_token`
(**KHÔNG** detach) — nhưng teacher ở `eval()` và không nằm trong optimizer nên gradient
qua `tea_token` cũng không cập nhật gì.

### (4) `gan_loss` — Loss để CẬP NHẬT Discriminator — [`dist_train_cakd.py:136-139`](dist_train_cakd.py)

```python
gan_loss = 0.5 * (gan_criterion(pred_real.detach(), True)
                + gan_criterion(pred_fake,         False))
```

$$\boxed{\;\mathcal{L}_{D} = \tfrac{1}{2}\Big[\ \ell\big(D(\bar A^{(1)}_T),\,1\big)
\;+\; \ell\big(D(A^{qk}_S),\,0\big)\ \Big]\;}$$

`pred_real` và `pred_fake` đều dựa trên input `.detach()` ⇒ backward chỉ chạm trọng số $D$,
KHÔNG chạm student. Đây là bước "dạy giám khảo phân biệt thật/giả".

---

## 4. Loss tổng của Student & lịch huấn luyện

### 4.1 Loss tổng — [`dist_train_cakd.py:146-151`](dist_train_cakd.py)

```python
loss = cls_loss + min(max(epoch - 25, 0) / 50.0, 0.2) * 1.0 * (
       pca_loss
     + gl_loss
     + 0.05 * gan_criterion(pred_real.detach(), True)
     + gan_criterion(pred_fake, True))
```

$$\boxed{\;\mathcal{L} = \mathcal{L}_{cls}
\;+\; \lambda(e)\cdot\Big(\,
\mathcal{L}_{pca} + \mathcal{L}_{gl}
+ 0.05\cdot\ell\big(D(\bar A^{(1)}_T)^{\text{det}},\,1\big)
+ \ell\big(D(A^{qk}_S),\,1\big)
\,\Big)\;}$$

$$\lambda(e) = \min\!\Big(\frac{\max(e-25,\,0)}{50},\ 0.2\Big)$$

> **Số hạng "loss chết":** $0.05\cdot\ell\big(D(\bar A^{(1)}_T)^{\text{det}},1\big)$ dùng
> `pred_real.detach()` ⇒ **gradient = 0**, không huấn luyện student (chỉ là hằng số cộng vào).
> Chỉ $\ell\big(D(A^{qk}_S),1\big)$ — student ép $D$ **tưởng attention của mình là THẬT** —
> mới thực sự tác động lên student (đây là số hạng generator non-saturating).

### 4.2 Hai bước tối ưu mỗi batch — [`dist_train_cakd.py:153-178`](dist_train_cakd.py)

1. **Cập nhật D:** `d_optimizer.zero_grad(); gan_loss.backward(retain_graph=True); d_optimizer.step()`
   (`retain_graph=True` để giữ đồ thị cho backward `loss` kế tiếp).
2. **Cập nhật Student:** `optimizer.zero_grad(); loss.backward(); optimizer.step()`
   (nhánh AMP dùng `scaler.scale(loss).backward()`).
3. **Teacher đóng băng:** `teacher.eval()` ([`dist_train_cakd.py:66`](dist_train_cakd.py)), không nằm trong optimizer nào.

`d_optimizer` có `lr = 0.01 × args.lr` ([`dist_train_cakd.py:486`](dist_train_cakd.py)) → D học chậm hơn student 100× (tránh D "quá mạnh" phá cân bằng đối kháng).

### 4.3 Lịch $\lambda(e)$ theo epoch (mặc định `--epochs 90`)

| Epoch $e$ | $\lambda(e)$ | Ý nghĩa |
|---|---|---|
| 0 – 25 | 0 | chỉ $\mathcal{L}_{cls}$ (tắt distill & GAN) — "khởi động chậm", tránh làm hỏng student sớm |
| 26 – 34 | 0.02 → 0.18 (tăng tuyến tính, bước 0.02/epoch) | bật distill + GAN, tăng dần ảnh hưởng |
| 35 – 89 | 0.2 (chạm trần) | distill + GAN chạy hết công suất |

Ví dụ: $\lambda(30)=\min(5/50,0.2)=0.1$; $\lambda(40)=\min(15/50,0.2)=\min(0.3,0.2)=0.2$.

---

## 5. So sánh trực tiếp: công thức bài báo ↔ code

| Thành phần | Bài báo (arXiv:2207.05273) | Code (thực tế) |
|---|---|---|
| **Loss tổng** | $\mathcal{L}_{total} = (\mathcal{L}_{proj1} + \mathcal{L}_{proj2}) + \lambda\,\mathcal{L}_{MVG}$ *(Eq. 10)* | $\mathcal{L} = \mathcal{L}_{cls} + \lambda(e)\,(\mathcal{L}_{pca} + \mathcal{L}_{gl} + 0.05\,\ell(D(\bar A^{(1)}_T)^{det},1) + \ell(D(A^{qk}_S),1))$ |
| **Phân loại** | *không nêu* | $\mathcal{L}_{cls} = \mathrm{CE}(z_S, y)$ |
| **Attention** (proj1 / pca) | $\lVert Attn_T - PCAttn_S\rVert^2 + \lVert V_TV_T/\sqrt d - V_SV_S/\sqrt d\rVert^2$ *(Eq. 4)* | $0.2\,\mathrm{MSE}(A^{qk}_S, \bar A^{(1)}_T) + 0.05\,\mathrm{MSE}(A^{vv}_S, \bar A^{(2)}_T)$ |
| **Feature** (proj2 / gl) | $\lVert h_T - h'_S\rVert^2$ *(Eq. 6)* | $\mathrm{MSE}(z_S,z_T) + \mathrm{MSE}(t_S,t_T) + 0.05\,\mathrm{MSE}(f_S,f_T)$ |
| **Discriminator** (MAD) | $\frac1m\sum[-\log D(h_T) - \log(1-D(h'_S))]$ *(Eq. 8)* | $\frac12[\ell(D(\bar A^{(1)}_T),1) + \ell(D(A^{qk}_S),0)]$ — trên **attention**, BCE |
| **Generator** (MVG) | $\frac1m\sum \log(1-D(h'_S))$ *(Eq. 9)* | $\ell(D(A^{qk}_S),1)$ — non-saturating $-\log D$ |
| **Trọng số $\lambda$** | hằng số | $\lambda(e) = \min(\max(e-25,0)/50,\ 0.2)$ |

**Khác biệt chính mỗi dòng:** (loss tổng) thêm $\mathcal{L}_{cls}$ + λ theo epoch · (attention) có hệ số 0.2/0.05, student trước-softmax vs teacher sau-softmax · (feature) khớp 3 thứ thay vì 1, logits dùng MSE không KL · (D/G) chạy trên attention map thay vì feature $h$, dùng BCE thay vì log thuần.

---

## 6. Chỗ bài báo SAI / THIẾU so với code

Bài báo (Eq. 4–10) — $\mathcal{L}_{total} = (\mathcal{L}_{proj1} + \mathcal{L}_{proj2}) + \lambda\,\mathcal{L}_{MVG}$ — lệch code ở 7 điểm:

1. **Thiếu $\mathcal{L}_{cls}$ và lịch $\lambda(e)$.** Code có CrossEntropy (thành phần chính) và $\lambda$
   **thay đổi theo epoch** (warmup từ epoch 25, trần 0.2); bài báo coi $\lambda$ là hằng số.

2. **Thiếu hệ số thật:** code dùng **0.2 / 0.05** trong $\mathcal{L}_{pca}$, **1 / 1 / 0.05** trong
   $\mathcal{L}_{gl}$, và **0.05** trước số hạng adversarial-real.

3. **$\mathcal{L}_{gl}$ rộng hơn $\mathcal{L}_{proj2}$.** Bài báo chỉ khớp feature $\lVert h_T - h_S'\rVert^2$.
   Code khớp **3 thứ**: logits (bằng **MSE**, không phải KL), class token, và feature.

4. **Adversarial chạy trên ATTENTION MAP, không phải feature $h$.** Bài báo cho $D$ phân biệt
   $h_T$ vs $h_S'$ (Eq. 8–9). Code cho $D$ (PatchGAN) phân biệt **attention** teacher
   $\bar A^{(1)}_T$ vs student $A^{qk}_S$, đầu vào (B, 1, 196, 196).

5. **GAN là vanilla (BCEWithLogits), không phải $\log(1-D)$ thuần.** Số hạng generator trong code
   là $\ell(D(A^{qk}_S),1)$ — bản "non-saturating" $-\log D$, ổn định hơn công thức
   $\mathcal{L}_{MVG} = \frac{1}{m}\sum \log(1-D(h_S'))$ của bài báo.

6. **Số hạng adversarial-real trong loss student là "loss chết"** (gradient = 0 do `pred_real.detach()`).

7. **Bất đối xứng softmax:** student so **điểm trước softmax** ($QK^\top/\sqrt d$) với **xác suất sau
   softmax** của teacher — khác thang đo. Đây có thể là lý do công thức attention trong bài báo trông "không khớp".

---

*Nguồn code: `train_one_epoch()` trong [`dist_train_cakd.py`](dist_train_cakd.py) dòng 49–199 (loss & GAN ở 118–178);
`ResNet_CAKD._forward_impl` + `Attention.forward` + `GLProj` trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py) (dòng 138–222, 631–666);
`VisionTransformer.forward` + `Encoder.forward` trong [`cakd_modified_files/vision_transformer.py`](cakd_modified_files/vision_transformer.py) (dòng 193–209, 351–374).*
