# CAKD — Công thức loss CHUẨN theo code

> Đối chiếu với bài báo *Cross-Architecture Knowledge Distillation* (arXiv:2207.05273).
> Các công thức dưới đây được viết lại cho **đúng 100% với code thực tế** trong
> [`dist_train_cakd.py`](dist_train_cakd.py) (loss) và [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py) (student).
> Những chỗ bài báo viết sai/thiếu được liệt kê ở cuối.

---

## 1. Ký hiệu (bám sát biến trong code)

| Ký hiệu | Biến trong code | Shape | Ghi chú |
|---|---|---|---|
| $z_S$ | `output` | (B, 1000) | logits student |
| $A^{qk}_S$ | `attn_weights[0]` | (B, 196, 196) | $\frac{1}{H}\sum_h \frac{Q_S K_S^\top}{\sqrt d}$ — **trước softmax** |
| $A^{vv}_S$ | `attn_weights[1]` | (B, 196, 196) | $\frac{1}{H}\sum_h \frac{V_S V_S^\top}{\sqrt d}$ — **trước softmax** |
| $f_S$ | `proj_feat` (vit_feat) | (B, 196, 768) | đầu ra `gl_proj` (GLProj) |
| $t_S$ | `proj_token` (cls_token) | (B, 768) | đầu ra `cls_proj` |
| $z_T$ | `tea_logits` | (B, 1000) | detach |
| $\bar A^{(1)}_T$ | `tea_attn_weights[2][:,1:,1:]` | (B, 196, 196) | attention teacher (bỏ CLS), **sau softmax**, detach |
| $\bar A^{(2)}_T$ | `tea_attn_weights[3][:,1:,1:]` | (B, 196, 196) | detach |
| $f_T$ | `tea_feat` | (B, 196, 768) | detach |
| $t_T$ | `tea_token` | (B, 768) | **không** detach |
| $D(\cdot)$ | `discriminator` (`NLayerDiscriminator`, PatchGAN) | — | đầu vào là attention map (B, 1, 196, 196) |

Quy ước:

- $\mathrm{MSE}(a,b) = \frac{1}{n}\lVert a-b\rVert_2^2$ — trung bình bình phương sai lệch (`nn.MSELoss`).
- $\ell(p, c) = \mathrm{BCEWithLogits}(p,\; c\cdot\mathbf{1})$, với $c=1$ (real) hoặc $c=0$ (fake).
  `GANLoss()` mặc định `gan_mode='vanilla'` → **BCEWithLogitsLoss** (KHÔNG phải LSGAN).

---

## 2. Các hàm loss

### (1) Phân loại — `dist_train_cakd.py:87`

$$\mathcal{L}_{cls} = \mathrm{CE}(z_S,\, y) \qquad (\text{có label smoothing})$$

### (2) PCA loss — khớp attention — `dist_train_cakd.py:90`

$$\mathcal{L}_{pca} = 0.2\cdot\mathrm{MSE}\!\big(A^{qk}_S,\ \bar A^{(1)}_T\big)
\;+\; 0.05\cdot\mathrm{MSE}\!\big(A^{vv}_S,\ \bar A^{(2)}_T\big)$$

### (3) GL loss — khớp logits + token + feature — `dist_train_cakd.py:92`

$$\mathcal{L}_{gl} = \mathrm{MSE}(z_S,\, z_T)
\;+\; \mathrm{MSE}(t_S,\, t_T)
\;+\; 0.05\cdot\mathrm{MSE}(f_S,\, f_T)$$

(logits: hệ số 1 · class token: hệ số 1 · patch feature: hệ số 0.05)

### (4) Loss của Discriminator (cập nhật D) — `dist_train_cakd.py:95`

$$\mathcal{L}_{D} = \tfrac{1}{2}\Big[\ \ell\big(D(\bar A^{(1)}_T),\,1\big)
\;+\; \ell\big(D(A^{qk}_S),\,0\big)\ \Big]$$

Cả hai đầu vào đều `.detach()` → chỉ cập nhật trọng số $D$.

### (5) Loss tổng của Student — `dist_train_cakd.py:102`

$$\mathcal{L} = \mathcal{L}_{cls}
\;+\; \lambda(e)\cdot\Big(\,
\mathcal{L}_{pca} + \mathcal{L}_{gl}
+ 0.05\cdot\ell\big(D(\bar A^{(1)}_T)^{\text{det}},\,1\big)
+ \ell\big(D(A^{qk}_S),\,1\big)
\,\Big)$$

$$\lambda(e) = \min\!\Big(\frac{\max(e-25,\,0)}{50},\ 0.2\Big)$$

> Số hạng $0.05\cdot\ell\big(D(\bar A^{(1)}_T)^{\text{det}},1\big)$ có `pred_real` đã `.detach()`
> ⇒ **gradient = 0**, không huấn luyện student. Chỉ $\ell\big(D(A^{qk}_S),1\big)$
> (student ép D tưởng attention của mình là THẬT) mới thực sự tác động lên student.

### (6) Tối ưu 2 bước mỗi batch

1. `d_optimizer`: cập nhật $D$ bằng $\mathcal{L}_D$ (`backward(retain_graph=True)`).
2. `optimizer`: cập nhật Student bằng $\mathcal{L}$.
3. Teacher **đóng băng** (`eval()`, không nằm trong optimizer).

`d_optimizer` có `lr = 0.01 × lr_student` (D học chậm hơn 100×).

### Lịch $\lambda$ theo epoch (mặc định 90 epoch)

| Epoch | $\lambda(e)$ | Ý nghĩa |
|---|---|---|
| 0 – 25 | 0 | chỉ $\mathcal{L}_{cls}$ (tắt distill & GAN) |
| 26 – 34 | 0.02 → 0.18 | bật distill, tăng dần |
| 35 – 89 | 0.2 (trần) | distill + GAN chạy hết công suất |

---

## 3. So sánh trực tiếp: công thức bài báo ↔ code

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

## 4. Chỗ bài báo SAI / THIẾU so với code

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

*Nguồn code: `train_one_epoch()` trong [`dist_train_cakd.py`](dist_train_cakd.py) dòng 66–102;
`ResNet_CAKD._forward_impl` và `Attention.forward` trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py).*
