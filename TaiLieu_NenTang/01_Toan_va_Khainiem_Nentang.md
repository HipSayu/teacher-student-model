# 01 — Toán & khái niệm nền tảng (cho CAKD)

Mọi công thức (1)–(10) trong bài báo đều ghép từ các "viên gạch" dưới đây. Mỗi mục: **trực giác → công thức → đầu vào/đầu ra → liên hệ CAKD**.

---

## 1. Tensor & shape — ngôn ngữ chung

**Tensor** = mảng số nhiều chiều. Học sâu chỉ là xếp các phép toán tensor lại.

- Số vô hướng (scalar): `()` — ví dụ giá trị loss.
- Véc-tơ (vector): `(D,)` — ví dụ một embedding 768 chiều.
- Ma trận (matrix): `(M, N)`.
- Tensor 4 chiều ảnh: `(B, C, H, W)`.
- Tensor 3 chiều chuỗi: `(B, N, D)`.

```
Ảnh RGB 1 tấm:        (3, 224, 224)        3 kênh màu, 224×224 điểm ảnh
Một lô B ảnh:         (B, 3, 224, 224)
Chuỗi token ViT:      (B, 197, 768)        197 token, mỗi token 768 chiều
```

> **Vì sao quan trọng:** KD xuyên kiến trúc khó *chính vì* hai shape `(B,C,H,W)` của CNN và `(B,N,D)` của Transformer khác nhau. Hai bộ chiếu PCA/GL của CAKD tồn tại để "dịch" giữa hai shape này.

---

## 2. Tích vô hướng (dot product) & nhân ma trận

**Tích vô hướng** hai véc-tơ `a, b ∈ R^D`:

```
a · b = Σ_i a_i b_i        (một số)
```

Ý nghĩa: đo mức "đồng hướng" giữa hai véc-tơ. Đây là hạt nhân của **attention** (so độ giống giữa query và key) và của **mọi lớp Linear**.

**Nhân ma trận** `A (M×K) · B (K×N) = C (M×N)`:

```
C[i,j] = Σ_k A[i,k] · B[k,j]       (yêu cầu: số cột A = số hàng B)
Input:  A=(M,K), B=(K,N)   →   Output: C=(M,N)
```

> Liên hệ CAKD: `Q K^T` với `Q=(N,d)`, `K=(N,d)` → `K^T=(d,N)` → kết quả `(N,N)` chính là **bản đồ chú ý** (mọi patch so với mọi patch).

---

## 3. Lớp tuyến tính (Linear / Fully-Connected, FC)

Phép biến đổi affine: `y = xW^T + b`.

```
Input:  x = (B, D_in)
Trọng số: W = (D_out, D_in),  b = (D_out,)
Output: y = (B, D_out)
```

Trực giác: trộn tuyến tính các đặc trưng đầu vào thành đặc trưng mới. Là khối cơ bản của MLP head, của projector GL (16 lớp FC), của discriminator.

> Liên hệ CAKD: **GL projector** = các lớp FC `1024 → 768` (đưa kênh CNN về số chiều embedding của ViT). **PCA projector** dùng một Linear `to_qkv` để sinh Q,K,V.

---

## 4. Hàm kích hoạt: ReLU, GELU, LeakyReLU, Sigmoid

Phi tuyến giúp mạng học quan hệ phức tạp (nếu chỉ có Linear xếp chồng thì vẫn chỉ là một Linear).

```
ReLU(x)      = max(0, x)                      ← phổ biến trong CNN/ResNet
LeakyReLU(x) = x nếu x>0, ngược lại 0.2x      ← dùng trong Discriminator (tránh "chết" nơ-ron)
GELU(x)      ≈ x · Φ(x)  (Φ: CDF chuẩn)       ← dùng trong MLP của Transformer
Sigmoid(x)   = 1/(1+e^{-x}) ∈ (0,1)           ← biến logit thành xác suất (GAN, nhị phân)
```

Input/Output: giữ nguyên shape (áp theo từng phần tử).

---

## 5. Softmax — biến điểm số thành phân phối xác suất

```
softmax(z)_i = e^{z_i} / Σ_j e^{z_j}          Σ_i softmax(z)_i = 1
Input: z=(..., N)   →   Output: cùng shape, mỗi hàng cộng lại = 1
```

Trực giác: "bình thường hóa" một dãy điểm số thành tỉ trọng. Dùng ở: (a) đầu ra phân loại, (b) **chuẩn hóa attention** `softmax(QK^T/√d)`.

> Liên hệ CAKD: trong code, `attn_qk = softmax(QK^T/√d)` rồi nhân `V` để ra đầu ra attention; nhưng phần **so khớp KD** lại dùng **điểm số trước softmax** `QK^T/√d` (xem file 03 & phần phân tích).

---

## 6. Hàm mất mát (loss): Cross-Entropy & MSE

### 6.1 Cross-Entropy (CE) — cho phân loại

```
CE(p, y) = − Σ_c y_c · log(p_c)        p = softmax(logits), y = nhãn one-hot
Input: logits=(B, num_classes), target=(B,)   →   Output: 1 số (trung bình theo batch)
```

Trực giác: phạt nặng khi mô hình tự tin nhưng sai. Là loss giám sát chính của bộ phân loại.

> Liên hệ CAKD: `cls_loss = CrossEntropy(output, target)` — mạch học từ nhãn thật của học sinh.

### 6.2 Mean Squared Error (MSE) — cho hồi quy/khớp đặc trưng

```
MSE(a, b) = (1/n) Σ_i (a_i − b_i)²
Input: hai tensor cùng shape   →   Output: 1 số
```

Trực giác: ép hai tensor **giống nhau từng phần tử**. Là "keo dán" của hầu hết KD theo đặc trưng.

> Liên hệ CAKD: `pca_loss`, `gl_loss` (các công thức 4 & 6) đều là MSE giữa đặc trưng/bản đồ chú ý của học sinh và giáo viên.

### 6.3 Binary Cross-Entropy (BCE) — cho nhị phân/GAN

```
BCE(p, t) = − [ t·log(p) + (1−t)·log(1−p) ]       t ∈ {0,1}
```

> Liên hệ CAKD: `GANLoss` dùng `BCEWithLogitsLoss` cho discriminator (thật=1, giả=0) — công thức (8)–(9).

---

## 7. Độ tương đồng cosin (cosine similarity)

```
cos(a, b) = (a · b) / (||a|| · ||b||) ∈ [−1, 1]
```

Trực giác: chỉ quan tâm **góc/hướng**, bỏ qua độ lớn. =1 cùng hướng, =0 vuông góc, =−1 ngược hướng.

> Liên hệ CAKD: bài báo dùng cosine để đo **"khả năng chuyển giao tri thức"** (Hình 1b, Hình 3b). Cùng kiến trúc ≈ 0,6–0,7; xuyên kiến trúc < 0,55. Sau khi thêm PCA+GL, cosine tăng vọt — bằng chứng hai bộ chiếu thu hẹp khoảng cách kiến trúc.

---

## 8. Ma trận Gram — quan hệ "cái này với cái kia"

Với `X = (N, d)`, ma trận Gram `G = X X^T = (N, N)`; `G[i,j] = x_i · x_j`.

Trực giác: tóm tắt **mối quan hệ tương hỗ** giữa các véc-tơ, bỏ qua vị trí tuyệt đối. Hay dùng trong style transfer và KD.

> Liên hệ CAKD: số hạng thứ 2 của công thức (4) `||V_T V_T^T/√d − V_S V_S^T/√d||²` chính là khớp **Gram của Value** giữa giáo viên và học sinh.

---

## 9. Gradient Descent & SGD — cách mạng "học"

Mạng có tham số `θ`. Ta muốn giảm loss `L(θ)`. **Gradient** `∇L` chỉ hướng tăng nhanh nhất → đi ngược lại:

```
θ ← θ − η · ∇L(θ)            η = learning rate (tốc độ học)
```

- **SGD** (Stochastic GD): tính gradient trên một *mini-batch* thay vì toàn bộ dữ liệu → nhanh, nhiễu có lợi.
- **Momentum**: cộng dồn quán tính để vượt vùng phẳng/dao động.
- **Weight decay**: phạt trọng số lớn (chống overfit).

> Liên hệ CAKD: dùng SGD, momentum 0,9, weight decay `1e-4`. Học sinh và Discriminator có **hai optimizer riêng** (D học chậm hơn, lr = 0,01×lr).

### Backpropagation (lan truyền ngược)

Quy tắc chuỗi (chain rule) để tính `∇L` theo *mọi* tham số một cách hiệu quả, lan từ loss ngược về đầu vào. `loss.backward()` trong PyTorch chính là bước này.

> Liên hệ CAKD: `.detach()` được dùng để **chặn gradient** chảy vào giáo viên (đóng băng) hoặc chặn nhánh không mong muốn khi cập nhật luân phiên Generator/Discriminator.

---

## 10. Chuẩn hóa: BatchNorm vs LayerNorm

Giữ phân phối kích hoạt ổn định để huấn luyện nhanh và bền.

```
chuẩn hóa: x̂ = (x − μ) / sqrt(σ² + ε),  rồi  y = γ·x̂ + β   (γ,β học được)
```

- **BatchNorm (BN):** thống kê μ,σ theo **chiều batch** cho mỗi kênh. Chuẩn của **CNN/ResNet**. Input `(B,C,H,W)`.
- **LayerNorm (LN):** thống kê theo **chiều đặc trưng** của *từng* mẫu. Chuẩn của **Transformer**. Input `(B,N,D)`.

> Liên hệ CAKD: ResNet học sinh dùng BN; ViT giáo viên dùng LN. Đây cũng là một biểu hiện "hai kiến trúc khác gen".

---

## 11. Dropout — chính quy hóa ngẫu nhiên

Khi huấn luyện, **tắt ngẫu nhiên** một tỉ lệ phần tử (đặt = 0) → mạng không phụ thuộc quá mức vào một vài nơ-ron, bớt overfit. Khi suy luận thì tắt dropout.

> Liên hệ CAKD: GL projector dùng dropout để "giảm tính toán và cải thiện độ bền vững".

---

## 12. Embedding & "token"

**Embedding** = biến một đối tượng (patch ảnh, từ, lớp) thành một **véc-tơ số** học được, sống trong không gian `D` chiều nơi "gần nhau = giống nhau".

- **Patch embedding (ViT):** mỗi mảnh ảnh 16×16×3 → một véc-tơ 768 chiều.
- **Class token:** một véc-tơ học được, ghép thêm vào chuỗi để "gom" thông tin toàn ảnh cho phân loại.
- **Positional embedding:** cộng thêm thông tin vị trí (vì attention vốn không biết thứ tự).

> Liên hệ CAKD: giáo viên trả về `cls_token (B,768)` và `feats (B,196,768)`; học sinh có `cls_proj` để tạo một "class token" tương ứng nhằm khớp.

---

## 13. EMA (Exponential Moving Average) của trọng số

Giữ một bản sao trọng số "mượt" cập nhật chậm:

```
θ_ema ← decay · θ_ema + (1 − decay) · θ        (decay ~ 0,9999)
```

Trực giác: trung bình hóa nhiều bước → mô hình ổn định hơn, thường chính xác hơn lúc test.

> Liên hệ CAKD: code có tùy chọn `model_ema` cho học sinh.

---

## Tóm tắt — "từ điển phép toán" của CAKD

| Khối toán | Xuất hiện ở | Công thức CAKD |
|---|---|---|
| Nhân ma trận, softmax | attention | (2), (3) |
| MSE | khớp attention/feature | (4), (6) |
| Cross-Entropy | phân loại học sinh | `cls_loss` |
| BCE/Sigmoid | GAN | (8), (9) |
| Ma trận Gram | khớp Value | (4) số hạng 2 |
| Cosine | đo chuyển giao | Hình 1b/3b |
| SGD + backprop + detach | huấn luyện luân phiên | (10), Thuật toán 1 |

→ Tiếp theo: `02_CNN.md` để hiểu **học sinh** sinh ra `h_S` như thế nào.
