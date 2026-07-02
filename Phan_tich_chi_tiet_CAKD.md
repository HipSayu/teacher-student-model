# Phân tích chi tiết: Cross-Architecture Knowledge Distillation (CAKD)

> **Bài báo:** *Cross-Architecture Knowledge Distillation* — Yufan Liu, Jiajiong Cao, Bing Li, Weiming Hu, Jingting Ding, Liang Li. **ACCV 2022**, trang 3396–3411. (arXiv:2207.05273v2)
>
> **Tài liệu này** giải thích cặn kẽ *từng công thức (1)–(10)*, *luồng hoạt động* của khung, *ý nghĩa* của từng thành phần, và **ánh xạ trực tiếp công thức ↔ code** trong project đi kèm (`CAKD/`).

---

## Mục lục

1. [Bức tranh tổng thể trong 60 giây (TL;DR)](#1-bức-tranh-tổng-thể-trong-60-giây-tldr)
2. [Bối cảnh & bài toán: vì sao chưng cất Transformer → CNN lại khó](#2-bối-cảnh--bài-toán-vì-sao-chưng-cất-transformer--cnn-lại-khó)
3. [Kiến trúc tổng thể & luồng dữ liệu](#3-kiến-trúc-tổng-thể--luồng-dữ-liệu)
4. [Bảng ký hiệu (notation) — đối chiếu paper ↔ code](#4-bảng-ký-hiệu-notation--đối-chiếu-paper--code)
5. [Thành phần 1 — Bộ chiếu PCA: công thức (1)–(4)](#5-thành-phần-1--bộ-chiếu-pca-công-thức-14)
6. [Thành phần 2 — Bộ chiếu GL: công thức (5)–(6)](#6-thành-phần-2--bộ-chiếu-gl-công-thức-56)
7. [Thành phần 3 — Huấn luyện bền vững đa khung nhìn: công thức (7)–(9)](#7-thành-phần-3--huấn-luyện-bền-vững-đa-khung-nhìn-công-thức-79)
8. [Tổng hàm mất mát & quy trình tối ưu: công thức (10) + Thuật toán 1](#8-tổng-hàm-mất-mát--quy-trình-tối-ưu-công-thức-10--thuật-toán-1)
9. [Đọc code theo luồng — file by file](#9-đọc-code-theo-luồng--file-by-file)
10. [Những điểm code khác với paper (rất quan trọng)](#10-những-điểm-code-khác-với-paper-rất-quan-trọng)
11. [Thí nghiệm, kết quả & ablation](#11-thí-nghiệm-kết-quả--ablation)
12. [Kết luận & nhận xét](#12-kết-luận--nhận-xét)

---

## 1. Bức tranh tổng thể trong 60 giây (TL;DR)

**Mục tiêu:** Dạy một mạng **CNN nhỏ gọn** (học sinh, ví dụ ResNet50) bằng tri thức của một mạng **Transformer mạnh** (giáo viên, ví dụ ViT-B/16). CNN dễ triển khai/tăng tốc trên phần cứng; Transformer chính xác cao nhưng nặng. Ta muốn lấy cái tốt của cả hai.

**Trở ngại:** CNN và Transformer "hình thành đặc trưng" theo cách *hoàn toàn khác nhau*, nên các phương pháp KD truyền thống (vốn giả định giáo viên–học sinh *cùng kiến trúc*) ép học sinh bắt chước trực tiếp đặc trưng giáo viên thì **thất bại** (khả năng chuyển giao tri thức rất thấp).

**Ý tưởng cốt lõi:** Đừng bắt chước *trực tiếp*. Thay vào đó dựng **hai "bộ phiên dịch" (projector)** để đưa đặc trưng CNN sang đúng "ngôn ngữ" của Transformer rồi mới so khớp:

| Bộ chiếu | Đưa đặc trưng CNN vào không gian | Học sinh học được gì | Công thức |
|---|---|---|---|
| **PCA** (Partially Cross Attention) | không gian **chú ý** (attention) của Transformer | quan hệ **toàn cục** giữa các vùng ảnh | (1)–(4) |
| **GL** (Group-wise Linear) | không gian **đặc trưng** Transformer (theo từng "điểm ảnh"/patch) | khớp đặc trưng **theo từng vị trí**, gọn nhẹ | (5)–(6) |

**Cộng thêm:** một **lược đồ huấn luyện bền vững đa khung nhìn (MVG + GAN)** dùng học đối kháng để đặc trưng học sinh "giống thật" như giáo viên, chịu nhiễu tốt hơn — công thức (7)–(9).

**Khi suy luận (inference):** vứt bỏ toàn bộ projector + discriminator, **chỉ giữ lại CNN học sinh** → không tốn thêm chi phí. Đây là điểm đẹp nhất của phương pháp.

---

## 2. Bối cảnh & bài toán: vì sao chưng cất Transformer → CNN lại khó

### 2.1 Chưng cất tri thức (KD) là gì

KD (Hinton và cộng sự, 2015) dùng khung **giáo viên–học sinh**: một mô hình lớn/mạnh (teacher) đã huấn luyện xong, và một mô hình nhỏ (student) được huấn luyện để *bắt chước* teacher chứ không chỉ học từ nhãn. "Tri thức" có thể là:

- **Đầu ra mềm (soft logits):** phân phối xác suất của teacher (mềm hơn nhãn one-hot, chứa thông tin "lớp nào giống lớp nào").
- **Đặc trưng trung gian (hint/feature):** các bản đồ đặc trưng ở lớp giữa của teacher.

### 2.2 Mấu chốt: CNN và Transformer "nghĩ" khác nhau

Đây là Hình 1(a) trong bài. Sự khác biệt về **cách hình thành đặc trưng**:

- **CNN:** đặc trưng trung gian là một khối `c × h' × w'` — tức **c kênh** của các **bản đồ đặc trưng** không gian `h'×w'`. Tích chập có *thiên kiến cục bộ* (local inductive bias): mỗi nơ-ron chỉ nhìn một vùng lân cận nhỏ.
- **Transformer:** chia ảnh thành **N mảnh (patch)**; đặc trưng là **N véc-tơ**, mỗi véc-tơ `3hw` chiều. Cơ chế **self-attention** cho phép mỗi patch "nhìn thấy" *mọi patch khác* ngay từ lớp đầu → nắm **quan hệ toàn cục**.

> Hệ quả: bản đồ đặc trưng CNN (dạng lưới không gian) và chuỗi token Transformer (dạng N véc-tơ) **không cùng định dạng, không cùng số chiều, không cùng ý nghĩa hình học**. Ép MSE trực tiếp giữa chúng là vô nghĩa.

### 2.3 "Khả năng chuyển giao tri thức" — bằng chứng định lượng (Hình 1(b))

Tác giả đo **độ tương đồng cosin** giữa đặc trưng học sinh (đã chiếu tuyến tính về cùng số chiều) và đặc trưng giáo viên:

- **Cùng kiến trúc** (CNN→CNN, T→T): cosin ≈ **0,6–0,7**.
- **Khác kiến trúc** (T→CNN): cosin **< 0,55**, thấp hơn hẳn.

→ Kết luận: khoảng cách giữa hai họ kiến trúc là *thật* và *đo được*. Cần một khung KD **mới** chuyên cho cảnh xuyên kiến trúc. Đây chính là động lực của CAKD.

### 2.4 Ba đóng góp chính

1. **Khung KD xuyên kiến trúc** với hai bộ chiếu **PCA** và **GL** để căn chỉnh không gian đặc trưng học sinh ↔ giáo viên, nâng khả năng chuyển giao.
2. **Lược đồ huấn luyện bền vững đa khung nhìn** để tăng độ ổn định/bền vững của học sinh.
3. **Thực nghiệm** vượt 14 phương pháp SOTA trên cả CIFAR (nhỏ) và ImageNet (lớn).

---

## 3. Kiến trúc tổng thể & luồng dữ liệu

Sơ đồ trong `img/framework.png`. Mô tả bằng lời theo **luồng đi của một ảnh**:

```
                 ┌───────────────────────── GIÁO VIÊN (Transformer, đóng băng) ─────────────────────────┐
   ảnh x ──MVG──►│ chia patch → Linear → [Transformer Block × L] → token cls → MLP Head → lớp          │
                 │                                   │ (lấy ở 2 block cuối) → Q_T,K_T,V_T  &  đặc trưng h_T│
                 └───────────────────────────────────┼──────────────────────────────────────────────────┘
                                                      │  so khớp trong 2 không gian
                 ┌────────────────────────────────────┼──────── HỌC SINH (CNN, cần huấn luyện) ──────────┐
   ảnh x ──MVG──►│ conv stem → layer1 → layer2 → layer3 ──► (đặc trưng h_S)                               │
                 │                                       ├─► PCA Projector ─► Q_S,K_S,V_S ─► attention map │──► L_proj1  (1)-(4)
                 │                                       ├─► GL  Projector ─► h'_S (N×3hw) ───────────────│──► L_proj2  (5)-(6)
                 │                                       └─► layer4 → avgpool → FC → lớp                   │──► L_cls (CrossEntropy)
                 │                                                         └─► cls_proj → token 768       │
                 └─────────────────────────────────────────────────────────────────────────────────────┘
                          attention map học sinh  ──►  Discriminator D  ◄── attention map giáo viên  ──► L_MVG / L_MAD  (7)-(9)
```

**Bốn khối chức năng:**

1. **PCA Projector** — biến đặc trưng CNN thành bộ ba `Q_S, K_S, V_S` rồi tính *bản đồ chú ý* của học sinh, để khớp với bản đồ chú ý của giáo viên ⇒ học sinh học **quan hệ toàn cục**.
2. **GL Projector** — ánh xạ đặc trưng CNN sang **không gian đặc trưng Transformer** theo từng patch (gọn nhẹ nhờ chia nhóm) ⇒ khớp đặc trưng **theo từng vị trí**.
3. **MVG (Multi-View Generator)** — sinh nhiều "khung nhìn" (biến đổi/nhiễu) của cùng một ảnh để gây nhiễu học sinh ⇒ ép học sinh bền vững.
4. **Discriminator D** — phân biệt đặc trưng giáo viên (thật) vs học sinh (giả). Học sinh đóng vai *generator* cố làm D nhầm ⇒ học đối kháng.

**Hai mạch giám sát song song của học sinh:**
- Mạch **gốc CNN**: `FC → CrossEntropy` với nhãn thật → giữ năng lực học **đặc trưng không gian cục bộ**.
- Mạch **chưng cất**: PCA + GL + GAN → bơm thêm **tri thức toàn cục** từ Transformer.

---

## 4. Bảng ký hiệu (notation) — đối chiếu paper ↔ code

| Ký hiệu (paper) | Ý nghĩa | Trong code | Kích thước (ảnh 224, ViT-B/16, ResNet50) |
|---|---|---|---|
| `x ∈ R^{3×H×W}` | ảnh đầu vào | `image` | `3×224×224` |
| `Θ_T` | giáo viên Transformer | `teacher = vit_b_16(...)` | — |
| `Θ_S` | học sinh CNN | `model = resnet50_cakd(...)` | — |
| `N` | số patch `= HW/(hw)` | — | `196 (=14×14)` |
| `h_T ∈ R^{N×3hw}` | đặc trưng giáo viên (chuỗi token patch) | `tea_feat` | `196×768` |
| `h_S ∈ R^{c×h'w'}` | đặc trưng học sinh (bản đồ CNN) | `x_3` (sau `layer3`) → `tmp` | code: `1024×196` |
| `Q_S,K_S,V_S` | Q/K/V của học sinh | bên trong `pca_proj` (`Attention`) | `16 đầu × 196 × 64` |
| `Q_T,K_T,V_T` | Q/K/V của giáo viên | bên trong MHA của ViT | `12 đầu × 197 × 64` |
| `Attn` | bản đồ chú ý | `attn_qk` / `dots_qk` | `196×196` |
| `h'_S ∈ R^{N×3hw}` | đặc trưng học sinh **sau GL** | `vit_feat = gl_proj(tmp)` | `196×768` |
| `D(·)` | bộ phân biệt đa khung nhìn | `discriminator` (`NLayerDiscriminator`) | vào: bản đồ attention 1 kênh |
| `λ` | hệ số cân bằng loss | lịch `min(max(epoch-25,0)/50, 0.2)` | 0 → 0.2 |

> ⚠️ **Lưu ý số chiều `h_S`:** paper viết `h_S ∈ R^{256×196}` (256 kênh), nhưng **code lấy đặc trưng ngay sau `layer3`** nên có **1024 kênh** (`R^{1024×196}`). Đây là một khác biệt cấu hình giữa paper và bản code công khai — xem [Mục 10](#10-những-điểm-code-khác-với-paper-rất-quan-trọng).

---

## 5. Thành phần 1 — Bộ chiếu PCA: công thức (1)–(4)

**Mục đích:** đưa đặc trưng CNN vào **không gian chú ý (attention) của Transformer**, để học sinh học được **quan hệ toàn cục** giữa các vùng ảnh — thứ mà self-attention của Transformer giỏi còn CNN thuần thì yếu.

### Công thức (1) — sinh Q, K, V cho học sinh

```
{Q_S, K_S, V_S} = Proj1(h_S)                                        (1)
```

- **Ý nghĩa:** lấy đặc trưng CNN `h_S` và "phiên dịch" thành ba ma trận Query/Key/Value — đúng bộ ba mà self-attention cần. Paper mô tả `Proj1` gồm **3 lớp conv 3×3**.
- **Trong code** (`resnet.py`, lớp `Attention`): dùng **một** `nn.Linear` sinh cả Q,K,V rồi tách 3:

```python
# resnet.py — class Attention
self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)   # dim=1024, inner_dim=16*64=1024
...
qkv = self.to_qkv(x).chunk(3, dim=-1)                      # tách thành Q, K, V
q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # 16 đầu
```

> Trước đó, đặc trưng CNN `x_3` dạng `(B, 1024, 14, 14)` được **làm phẳng & hoán vị** thành chuỗi `(B, 196, 1024)` để giống định dạng token của Transformer:
> ```python
> tmp = torch.reshape(x_3, (x_3.shape[0], x_3.shape[1], -1))  # (B,1024,196)
> tmp = tmp.permute((0,2,1))                                  # (B,196,1024)  ← 196 "token", mỗi token 1024 chiều
> ```

### Công thức (2) — tự chú ý của học sinh

```
Attn_S = softmax( Q_S (K_S)^T / sqrt(d) ) · V_S                     (2)
```

- **Ý nghĩa:** đây *đúng* là công thức self-attention chuẩn. `Q_S (K_S)^T` cho ma trận **196×196** đo độ "liên quan" giữa mọi cặp patch; chia `sqrt(d)` để ổn định số học (`d` = số chiều mỗi đầu = 64); `softmax` biến thành trọng số; nhân `V_S` để tổng hợp thông tin toàn ảnh. Học sinh tính `Attn_S` *y hệt cách giáo viên tính `Attn_T`* ⇒ giờ hai bên đã "cùng ngôn ngữ" và có thể so khớp.
- **Trong code:**

```python
# resnet.py — class Attention.forward
dots_qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # = Q_S K_S^T / sqrt(d), self.scale = d^-0.5
dots_vv = torch.matmul(v, v.transpose(-1, -2)) * self.scale   # = V_S V_S^T / sqrt(d)   (cho số hạng 2 của (4))
attn_qk = self.attend(dots_qk)                                # softmax
out = torch.matmul(attn_qk, v)                                # = Attn_S (đầu ra attention)
return self.to_out(out), dots_qk, dots_vv                     # ← TRẢ VỀ dots_qk, dots_vv (TRƯỚC softmax)
```

### Công thức (3) — "Partially Cross" Attention (mẹo bền vững)

```
PCAttn_S = softmax( g(Q_S) g(K_S)^T / sqrt(d) ) · g(V_S)
với  g(M(i,j)) = M_T(i,j) nếu p ≥ 0,5 ;  = M_S(i,j) nếu p < 0,5 ;   M ∈ {Q,K,V}     (3)
```

- **Ý nghĩa (chữ "Partially Cross"):** với xác suất `p` (phân phối đều), **thay ngẫu nhiên** một phần các phần tử trong `Q_S/K_S/V_S` của học sinh bằng phần tử tương ứng của **giáo viên** `Q_T/K_T/V_T`. Giống "dropout chéo": đôi khi học sinh được "mượn tạm" tham số giáo viên. Điều này (a) buộc học sinh tạo ra Q/K/V *tương thích* với của giáo viên, (b) tăng tính bền vững/đa dạng khi huấn luyện. Đây là lý do tên gọi **Partially Cross Attention**.

### Công thức (4) — hàm mất mát của PCA

```
L_proj1 = || Attn_T − PCAttn_S ||²₂  +  || V_T V_T^T / sqrt(d) − V_S V_S^T / sqrt(d) ||²₂     (4)
```

- **Số hạng 1** `|| Attn_T − PCAttn_S ||²`: ép **bản đồ chú ý** của học sinh giống của giáo viên ⇒ truyền *cấu trúc quan hệ toàn cục*.
- **Số hạng 2** `|| V_T V_T^T/√d − V_S V_S^T/√d ||²`: ép **ma trận Gram của Value** (quan hệ giữa các value với nhau) khớp nhau ⇒ ổn định thêm không gian biểu diễn, độc lập với riêng phần attention.
- **Trong code** (`dist_train_cakd.py`, `train_one_epoch`):

```python
pca_loss = 0.2 * mse_criterion(attn_weights[0], tea_attn_weights[2][:, 1:, 1:].detach()) \
         + 0.05 * mse_criterion(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())
#          └ student dots_qk  ┘   └ teacher QK map (block cuối) ┘   └ trọng số 0.2
#          + student dots_vv         teacher VV map (block cuối)       trọng số 0.05
```

  - `attn_weights[0]` = `attn_qk` (học sinh), `tea_attn_weights[2]` = bản đồ QK của **block Transformer cuối**.
  - `[:, 1:, 1:]` = **cắt bỏ class token** (hàng/cột đầu) để còn `196×196`, khớp số patch của học sinh.
  - `.detach()` = không cho gradient chảy ngược vào giáo viên (giáo viên bị đóng băng).
  - Hai trọng số `0.2` và `0.05` chính là hiện thực hóa "tỉ lệ" giữa hai số hạng trong (4).

> 🔎 **Tinh ý:** Code so khớp **ma trận điểm số trước softmax** `Q K^T/√d` (gọi `dots_qk`), *không* phải bản đồ sau softmax, và cũng không phải đầu ra `Attn=softmax(...)V` như chữ trong (4). Cách hiểu "attention map" ở đây = ma trận điểm số đã chia `√d`. Đây là chi tiết hiện thực rất đáng lưu ý khi đọc song song paper–code.

---

## 6. Thành phần 2 — Bộ chiếu GL: công thức (5)–(6)

**Mục đích:** đưa đặc trưng CNN sang **không gian đặc trưng Transformer** và khớp **theo từng "điểm ảnh"/patch**, trực tiếp giảm khác biệt *cách hình thành đặc trưng*.

### Công thức (5) — chiếu đặc trưng

```
h'_S = Proj2(h_S)                                                    (5)
```

- `h'_S ∈ R^{N×3hw}` được căn chỉnh để **cùng số chiều với `h_T`** (giáo viên). Với 224×224: `h_S ∈ R^{256×196}` (paper) → `h'_S ∈ R^{196×768}`.

### Vì sao cần "Group-wise" (theo nhóm)?

Để ánh xạ *từng patch riêng* từ không gian CNN sang không gian Transformer, về lý thuyết cần **196 lớp FC riêng** (mỗi lớp `256×768` tham số) → **rất tốn**. Giải pháp của tác giả: cho **một vùng lân cận 4×4 patch dùng chung một lớp FC** ⇒ chỉ còn **16 lớp FC** (vì lưới 14×14 chia thành các nhóm). Thêm **dropout** để nhẹ hơn và bền hơn.

- **Trong code** (`resnet.py`, lớp `GLProj`): đúng **16 lớp FC** khi `num_patch=196`:

```python
class GLProj(nn.Module):
    def __init__(self, src_dim=1024, tgt_dim=768, num_patch=196):
        if num_patch == 196: num_fc = 16        # 196 patch → 16 nhóm
        elif num_patch == 49: num_fc = 4
        ...
        for i in range(num_fc):
            layers[f"fc_layer_{i}"] = nn.Linear(src_dim, tgt_dim)   # mỗi nhóm 1 FC: 1024→768

    def forward(self, x):
        out = torch.zeros((x.shape[0], x.shape[1], self.tgt_dim)).cuda()
        for i in range(num_fc):
            out[:, idx[i], :] = self.layers[i](x[:, idx[i], :])      # FC i chỉ xử lý các patch thuộc nhóm i
        return out
```

  - `idx[i]` (= `idx_224_16_i`) là **danh sách chỉ số các patch** thuộc nhóm `i`. Các chỉ số này được gom theo **lân cận không gian 4×4** trên lưới 14×14 — đúng tinh thần "một lân cận 4×4 chung một FC". Ví dụ nhóm 0:
    ```python
    idx_224_16_0 = [0,1,2,3, 14,15,16,17, 28,29,30,31, 42,43,44,45]  # một mảng vuông góc trên-trái của lưới 14×14
    ```

### Công thức (6) — hàm mất mát của GL

```
L_proj2 = || h_T − h'_S ||²₂                                         (6)
```

- **Ý nghĩa:** MSE trực tiếp giữa đặc trưng giáo viên `h_T` và đặc trưng học sinh đã chiếu `h'_S`, *theo từng patch*. Sau khi GL đã đưa hai bên về cùng không gian/số chiều, phép so khớp này mới có nghĩa.
- **Trong code** (`gl_loss`, gộp 3 số hạng — xem chi tiết ở Mục 9):

```python
gl_loss = mse_criterion(output, tea_logits.detach())      # (a) khớp logits (KD kinh điển dạng MSE)
        + mse_criterion(proj_token, tea_token)            # (b) khớp class-token: cls_proj(cnn_token) ↔ token cls giáo viên
        + 0.05 * mse_criterion(proj_feat, tea_feat.detach())  # (c) ĐÚNG công thức (6): h'_S ↔ h_T
```

  - `proj_feat = vit_feat = gl_proj(tmp)` chính là `h'_S`; `tea_feat` chính là `h_T`. Số hạng **(c)** = công thức (6).
  - Code **bổ sung** (a) khớp logits và (b) khớp class token — đây là phần "thưởng thêm" ngoài (6), giúp truyền cả tri thức ở mức quyết định cuối.

---

## 7. Thành phần 3 — Huấn luyện bền vững đa khung nhìn: công thức (7)–(9)

**Mục đích:** vì khoảng cách kiến trúc lớn, học sinh dễ *bất ổn*. Dùng **học đối kháng (GAN)** để đặc trưng học sinh "không phân biệt được" với giáo viên dưới **nhiều khung nhìn nhiễu**.

### Công thức (7) — Bộ sinh đa khung nhìn (MVG)

```
x̃ = MVG(x) = Trans(x) nếu p ≥ 0,5 ;  = x nếu p < 0,5                 (7)
```

- **Ý nghĩa:** với xác suất ~0,5, áp một biến đổi ngẫu nhiên `Trans(·)` (đổi màu/jitter, crop ngẫu nhiên, xoay, che mảnh — *patch mask*, v.v.) lên ảnh; nếu không thì giữ nguyên. Mỗi ảnh do đó có nhiều "khung nhìn" → gây nhiễu có kiểm soát cho học sinh.
- **Trong code:** bản công khai **không** dùng một lớp `MVG` riêng; theo README, nhóm tác giả dùng **augmentor gốc của PyTorch** thay cho phần này (`RandomResizedCrop`, `RandAugment/AutoAugment`, `RandomErasing` trong `new_utils.ClassificationPresetTrain`, cùng `RandomMixup/RandomCutmix` trong `transforms.py`). Tức "đa khung nhìn" được hiện thực bằng pipeline tăng cường dữ liệu tiêu chuẩn.

### Công thức (8) — mất mát của Discriminator (cập nhật D)

```
L_MAD = (1/m) Σ_k [ −log D(h_T^(k)) − log(1 − D(h'_S^(k))) ]          (8)
```

- **Ý nghĩa:** đây là mục tiêu GAN kinh điển cho **bộ phân biệt** `D`: học để gán **1 (thật)** cho đặc trưng giáo viên `h_T`, **0 (giả)** cho đặc trưng học sinh `h'_S`. `m` = số mẫu.
- **Trong code:**

```python
gan_criterion = new_utils.GANLoss()        # BCEWithLogitsLoss (GAN "vanilla")
...
pred_real = discriminator(input_d_real)    # input_d_real = bản đồ attention GIÁO VIÊN
pred_fake = discriminator(input_d_fake.detach())  # input_d_fake = bản đồ attention HỌC SINH
gan_loss  = 0.5 * (gan_criterion(pred_real.detach(), True) + gan_criterion(pred_fake, False))
# True→nhãn 1 (thật) cho giáo viên ; False→nhãn 0 (giả) cho học sinh  ⇒ đúng (8)
```

  - Cập nhật D riêng bằng `d_optimizer` (learning rate = `0.01 × lr` của học sinh — D học chậm hơn để ổn định).

### Công thức (9) — mất mát đối kháng của học sinh (generator)

```
L_MVG = (1/m) Σ_k log(1 − D(h'_S^(k)))                                (9)
```

- **Ý nghĩa:** học sinh đóng vai **generator**, muốn **làm D nhầm** — tức ép `D(h'_S) → 1`. Tối thiểu (9) ⇒ phân phối đặc trưng học sinh tiến gần phân phối đặc trưng giáo viên.
- **Trong code** (số hạng nằm trong tổng loss của học sinh, dùng dạng **non-saturating** `−log D` thay cho `log(1−D)` cho ổn định gradient):

```python
# phần generator trong loss tổng của học sinh:
0.05 * gan_criterion(pred_real.detach(), True) + gan_criterion(pred_fake, True)
#                                                 └ ép D(student) → "thật" (True)  ⇒ đúng tinh thần (9)
```

> 🔎 **Tinh ý:** Theo paper, D phân biệt **đặc trưng** `h_T` vs `h'_S` (không gian đặc trưng Transformer). Trong code công khai, D lại nhận **bản đồ attention** (`input_d_real/fake` lấy từ attention map giáo viên/học sinh, 1 kênh). Khác biệt hiện thực — xem Mục 10.

---

## 8. Tổng hàm mất mát & quy trình tối ưu: công thức (10) + Thuật toán 1

### Công thức (10) — tổng loss của học sinh

```
L_total = (L_proj1 + L_proj2) + λ · L_MVG                             (10)
```

- `L_proj1` (PCA) + `L_proj2` (GL) = phần **chưng cất căn chỉnh đặc trưng**; `λ·L_MVG` = phần **đối kháng**. `λ` cân bằng các số hạng.
- **Trong code**, tổng loss được viết gọn và thêm **lịch trọng số khởi động (warm-up)**:

```python
cls_loss = criterion(output, target)            # CrossEntropy với nhãn thật (mạch CNN gốc)
loss = cls_loss \
     + min(max(epoch-25, 0)/50.0, 0.2) * 1.0 * ( pca_loss + gl_loss
                                                 + 0.05*gan_criterion(pred_real.detach(), True)
                                                 + gan_criterion(pred_fake, True) )
```

  - `cls_loss` (CrossEntropy) là phần **không có trong (10)** nhưng tối quan trọng: giữ cho học sinh vẫn học phân loại từ nhãn thật (mạch "đặc trưng cục bộ CNN").
  - **Hệ số `min(max(epoch-25,0)/50, 0.2)`** đóng vai trò `λ`: **0 trong 25 epoch đầu** (chỉ học phân loại để CNN "đứng vững"), rồi **tăng tuyến tính** tới **0,2** ở epoch 75, sau đó giữ nguyên. Đây là mẹo ổn định: bật chưng cất *từ từ* sau khi học sinh đã có nền tảng.
  - `pca_loss + gl_loss` = `L_proj1 + L_proj2`; hai số hạng GAN cuối = `λ·L_MVG`.

### Thuật toán 1 — quy trình huấn luyện (cập nhật luân phiên)

Mỗi vòng lặp (mỗi batch) cập nhật **hai nhóm tham số xen kẽ**:

```python
# 1) Cập nhật DISCRIMINATOR trước:
d_optimizer.zero_grad()
gan_loss.backward(retain_graph=True)     # ∇ của L_MAD (8)
d_optimizer.step()

# 2) Cập nhật HỌC SINH (Θ_S + Proj1 + Proj2):
optimizer.zero_grad()
loss.backward()                          # ∇ của L_total (10)
optimizer.step()
```

**Toàn cảnh quy trình:**

1. Dựng khung giáo viên–học sinh xuyên kiến trúc (`teacher = vit_b_16` đóng băng & `eval()`, `model = resnet50_cakd` huấn luyện).
2. Nhúng PCA (`pca_proj`) và GL (`gl_proj`) vào học sinh để chiếu đặc trưng sang không gian chú ý & đặc trưng của giáo viên.
3. Áp huấn luyện bền vững đa khung nhìn; cập nhật **luân phiên** thân chính (`Θ_S, Proj1, Proj2`) và `D`.
4. **Sau hội tụ:** **bỏ** `Proj1, Proj2, D`, **chỉ giữ CNN học sinh `Θ_S`** để suy luận → nhẹ, nhanh, thân thiện phần cứng.

---

## 9. Đọc code theo luồng — file by file

Cấu trúc project:

```
CAKD/
├── dist_train_cakd.py          # ★ Vòng huấn luyện CAKD (loss, optimizer, train loop)
├── dist_train_logits.py        # baseline: chỉ KD bằng logits
├── dist_train_student.py       # baseline: học sinh học chay (không KD)
├── new_utils.py                # GANLoss, Discriminator (NLayerDiscriminator), tiện ích train
├── transforms.py               # RandomMixup, RandomCutmix (một phần của "đa khung nhìn")
├── cakd_modified_files/
│   ├── resnet.py               # ★ ResNet_CAKD + Attention(PCA) + GLProj(GL)
│   ├── vision_transformer.py   # ★ ViT sửa để trả về attention map + token + feature
│   └── functional.py           # ★ sửa multi_head_attention để trả về [QK, VV] (trước softmax)
└── experiments/                # script chạy: run_baseline.sh / run_logits.sh / run_cakd.sh
```

> Vì sao phải **sửa torch gốc** (bước 2 trong README)? Mặc định `nn.MultiheadAttention` của PyTorch **không trả về** ma trận `QK^T` và `VV^T` thô. Tác giả chép đè `resnet.py`, `vision_transformer.py`, `functional.py` để "khoét" các đầu ra trung gian này ra cho KD.

### 9.1 `resnet.py` — học sinh `ResNet_CAKD`

Điểm mấu chốt: projector được gắn **ngay sau `layer3`** (nên đặc trưng có 1024 kênh), `layer4` vẫn chạy bình thường cho nhánh phân loại:

```python
# __init__:
self.layer3   = self._make_layer(block, 256, layers[2], stride=2)   # → inplanes = 1024
self.pca_proj = Attention(dim=self.inplanes, heads=16, dim_head=self.inplanes//16)  # PCA
self.gl_proj  = GLProj(src_dim=self.inplanes, tgt_dim=768, num_patch=196)           # GL
self.layer4   = self._make_layer(block, 512, layers[3], stride=2)
self.fc       = nn.Linear(512*block.expansion, num_classes)   # đầu phân loại
self.cls_proj = nn.Linear(512*block.expansion, 768)           # chiếu token CNN → 768 (khớp class-token ViT)

# _forward_impl:
x_3 = self.layer3(x)                                  # đặc trưng h_S: (B,1024,14,14)
tmp = x_3.reshape(B,1024,-1).permute(0,2,1)           # → (B,196,1024) dạng "token"
_, attn_qk, attn_vv = self.pca_proj(tmp)              # bản đồ chú ý học sinh (PCA)
attn_qk = attn_qk.sum(1)/num_heads                    # gộp 16 đầu → (B,196,196)
attn_vv = attn_vv.sum(1)/num_heads
vit_feat = self.gl_proj(tmp)                          # h'_S: (B,196,768) (GL)
x = self.layer4(x_3); x = self.avgpool(x)
cnn_token = torch.flatten(x,1)                        # (B,2048)
return self.fc(cnn_token), [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)
#       └ logits ┘          └ cho L_proj1 (4) ┘   └ h'_S (6)┘ └ token cho gl_loss(b)┘
```

→ Học sinh trả về **4 thứ**: `(logits, [attn_qk, attn_vv], vit_feat, cls_token_proj)` — khớp đúng với 4 tham số trong `train_one_epoch`.

### 9.2 `vision_transformer.py` + `functional.py` — giáo viên ViT

- `functional.py` (`_scaled_dot_product_attention`) được sửa để **trả về cả `[attn, attn_vv]` trước softmax**:
  ```python
  attn    = torch.bmm(q, k.transpose(-2,-1))   # QK^T (q đã chia sqrt(E)) → = QK^T/√d
  attn_vv = torch.bmm(v, v.transpose(-2,-1))   # VV^T
  attn_soft = softmax(attn, -1)                # softmax chỉ dùng nội bộ để ra output
  output = torch.bmm(attn_soft, v)
  return output, [attn, attn_vv]
  ```
- `Encoder.forward` lấy attention của **2 block cuối**:
  ```python
  for i in range(num_layers):
      if i < num_layers-2:   x = self.layers[i](x)                       # block thường
      elif i == num_layers-2: x, attn_weights_2 = self.layers[i](x, True)  # block kế cuối
      else:                   x, attn_weights_1 = self.layers[i](x, True)  # block cuối
  return self.ln(x), [attn_weights_2[0], attn_weights_2[1],   # [0],[1] = QK,VV kế cuối
                      attn_weights_1[0], attn_weights_1[1]]    # [2],[3] = QK,VV block cuối
  ```
- `VisionTransformer.forward` trả về **4 thứ** khớp với teacher trong train loop:
  ```python
  return x, attn_weights, cls_token, feats
  #      │   │             │          └ feats = h_T (B,196,768)  → tea_feat
  #      │   │             └ token lớp (B,768)                   → tea_token
  #      │   └ 4 bản đồ attention (QK/VV ×2 block)               → tea_attn_weights
  #      └ logits giáo viên                                      → tea_logits
  ```

### 9.3 `new_utils.py` — GANLoss & Discriminator

- `GANLoss(gan_mode='vanilla')` ⇒ dùng `BCEWithLogitsLoss`; gọi `gan_criterion(pred, True/False)` tự tạo nhãn 1/0 đúng kích thước.
- `NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3)` = **PatchGAN** nhận **bản đồ attention 1 kênh** (`(B,1,196,196)`), qua vài conv `stride=2` + `LeakyReLU` rồi `AdaptiveAvgPool2d(1)` → một điểm số thật/giả.

### 9.4 `dist_train_cakd.py` — ghép tất cả

Bản đồ "loss trong code ↔ công thức trong paper":

| Biến code | Công thức | Vai trò |
|---|---|---|
| `cls_loss` | (ngoài (10)) | CrossEntropy nhãn thật — mạch CNN gốc |
| `pca_loss` | **(4)** `L_proj1` | khớp bản đồ chú ý QK + Gram VV |
| `gl_loss` (số hạng `0.05*MSE(proj_feat, tea_feat)`) | **(6)** `L_proj2` | khớp đặc trưng patch `h'_S ↔ h_T` |
| `gl_loss` (số hạng `MSE(output, tea_logits)`) | — | KD logits (thưởng thêm) |
| `gl_loss` (số hạng `MSE(proj_token, tea_token)`) | — | khớp class-token |
| `gan_loss` | **(8)** `L_MAD` | cập nhật Discriminator |
| `gan_criterion(pred_fake, True)` trong `loss` | **(9)** `L_MVG` | học sinh làm D nhầm |
| `loss` | **(10)** `L_total` | tổng + lịch warm-up làm `λ` |

---

## 10. Những điểm code khác với paper (rất quan trọng)

Bản code công khai **không phải bản gốc đầy đủ** — README ghi rõ nhóm tác giả lược bỏ nhiều "phép tăng cường tùy biến" và *teacher tùy biến* vì lý do bảo mật. Do đó có vài khác biệt cần biết khi đọc song song:

1. **Số kênh `h_S`:** paper viết `256×196`, code dùng đặc trưng sau `layer3` = **`1024×196`**.
2. **`Proj1` (PCA):** paper nói "3 conv 3×3", code dùng **một `nn.Linear`** sinh Q,K,V (`to_qkv`).
3. **Partially-Cross (công thức 3):** mẹo *thay ngẫu nhiên Q/K/V học sinh bằng của giáo viên* **không xuất hiện** trong lớp `Attention` công khai — code chỉ tính attention thuần của học sinh rồi MSE với giáo viên.
4. **"Attention map" được khớp:** code khớp ma trận **trước softmax** `QK^T/√d` và `VV^T`, không phải đầu ra `softmax(...)V`.
5. **Đầu vào Discriminator:** paper mô tả D phân biệt **đặc trưng** `h_T`/`h'_S`; code cho D ăn **bản đồ attention** của giáo viên/học sinh.
6. **MVG (công thức 7):** không có lớp `MVG` riêng; "đa khung nhìn" được thay bằng **augmentor PyTorch chuẩn** (RandAugment, RandomErasing, Mixup, CutMix…).
7. **`gl_loss` mở rộng:** ngoài (6), code thêm khớp **logits** và **class-token** — đây là phần "cộng thêm" thực dụng.
8. **`λ` = lịch warm-up** `min(max(epoch-25,0)/50, 0.2)`, bật chưng cất sau epoch 25 (paper chỉ nói `λ` là hệ số cân bằng).

> Vì những lược bỏ này, README cũng nói thẳng: **độ chính xác tuyệt đối thấp hơn trong paper**, nhưng **mức cải thiện so với baseline thì vẫn rõ rệt** (xem dưới).

---

## 11. Thí nghiệm, kết quả & ablation

### 11.1 Thiết lập

- **Dữ liệu:** CIFAR (nhỏ) và ImageNet (lớn); tăng cường theo ví dụ chính thức của PyTorch.
- **Học sinh (CNN):** ResNet, MobileNetV2, Xception, EfficientNet. **Giáo viên (Transformer):** ViT, Swin.
- **Huấn luyện:** CIFAR 200 epoch (batch 64, lr 0,1, giảm ×0,1 ở epoch 100/150); ImageNet 120 epoch (batch 256, giảm ở 30/60/90). SGD, weight decay `1e-4`, momentum 0,9, 8 GPU. Mỗi cấu hình lặp 5 lần với seed khác nhau (đo độ ổn định).
- **So sánh:** 14 SOTA — Logits, FitNet, AT, IRG, RKD, CRD, OFD, ReviewKD, LONDON, AFD, AB, FT (nhóm CNN) + DeiT, MINILM (nhóm Transformer).

### 11.2 Kết quả chính (định tính từ bài báo)

- **Đứng đầu** trên cả CIFAR100 và ImageNet, vượt mọi phương pháp KD dựa trên CNN lẫn dựa trên Transformer.
- Chế độ **Transformer→CNN** cho mức tăng cao hơn các phương pháp CNN→CNN (**trung bình +2,7%** trên CIFAR100) ⇒ chứng tỏ KD cũ *không khai thác hết* giáo viên Transformer.
- Trên ImageNet, có CNN được Transformer dạy **vượt cả Transformer cùng mức FLOPs** (vd ResNet50x2 đạt 80,72% > ViT-B/32 78,29%, +1,03%) — mà lại **thân thiện phần cứng** hơn.
- Học sinh càng tốt khi **giáo viên càng mạnh** ⇒ Transformer là "giáo viên xuất sắc" tiềm năng.

### 11.3 Kết quả của bản code công khai (README)

**ImageNet, học sinh ResNet50:**

| Phương pháp | Top-1 | Top-5 |
|---|---|---|
| Baseline (ResNet50) | 73,82% | 91,97% |
| Logits (KD thường) | 74,48% | 92,29% |
| **CAKD (bài báo này)** | **76,21%** | **93,09%** |

→ CAKD **+2,39%** Top-1 so với baseline, và **+1,73%** so với KD logits thường — đúng như tuyên bố "mức cải thiện rõ rệt" dù tuyệt đối thấp hơn bản nội bộ.

### 11.4 Ablation (Hình 3 & Bảng 3)

- **Khái quát theo cặp:** nhiều cặp giáo viên–học sinh xuyên kiến trúc đều cải thiện so với baseline; giáo viên mạnh ⇒ học sinh mạnh.
- **Hiệu quả của 2 projector:** thêm PCA + GL ⇒ tăng độ chính xác lớn **và** kéo **độ tương đồng cosin** (khả năng chuyển giao) lên *cao hơn cả cảnh cùng kiến trúc* — đúng mục tiêu thu hẹp khoảng cách kiến trúc.
- **Hiệu quả huấn luyện bền vững đa khung nhìn:** đánh giá không nhiễu +0,2–0,4% Top-1; **đánh giá có nhiễu +hơn 1,0%** ⇒ tăng *độ bền vững với nhiễu* rõ rệt.
- **Tác vụ khác (Bảng 4):** áp thẳng cho phát hiện đối tượng, phân đoạn thể hiện (COCO, đo AP) và chống giả mạo khuôn mặt (CelebA-Spoof, đo EER) đều tốt hơn KD thường ⇒ khả năng khái quát ngoài phân loại.

---

## 12. Kết luận & nhận xét

### 12.1 Tóm lại bài báo làm gì

CAKD giải bài toán **chưng cất Transformer → CNN** bằng cách **không bắt chước trực tiếp** mà **phiên dịch** đặc trưng CNN sang hai không gian của Transformer:
- **PCA** (1)–(4): học **quan hệ toàn cục** qua không gian attention.
- **GL** (5)–(6): khớp **đặc trưng theo patch** một cách gọn nhẹ (16 FC theo nhóm).
- **MVG + GAN** (7)–(9): học **đối kháng đa khung nhìn** để bền vững.
- **Tổng** (10) + cập nhật luân phiên; **suy luận chỉ còn CNN** → "được Transformer dạy nhưng nhẹ như CNN".

### 12.2 Vì sao thiết kế này hợp lý

- Khớp ở **không gian trung gian đã căn chỉnh** né được vấn đề "khác định dạng/số chiều" của hai kiến trúc.
- Tách **hai loại tri thức**: quan hệ toàn cục (PCA) và biểu diễn theo vị trí (GL) — bổ sung cho nhau.
- GAN + đa khung nhìn xử lý **độ bất ổn** vốn lớn trong cảnh xuyên kiến trúc.
- Bỏ projector khi suy luận ⇒ **không tăng chi phí triển khai** — yếu tố thực dụng quan trọng cho thiết bị biên.

### 12.3 Lưu ý khi tái lập

- Phải **chép đè 3 file torch** (`resnet.py`, `vision_transformer.py`, `functional.py`) đúng đường dẫn `TORCHVISION_MODEL_PATH` / `TORCH_NN_PATH` thì mới lấy được attention/feature trung gian.
- Giáo viên dùng `vit_b_16` pretrained ImageNet (đóng băng, `eval()`); học sinh `resnet50_cakd`.
- Nhớ các **khác biệt paper↔code** ở [Mục 10](#10-những-điểm-code-khác-với-paper-rất-quan-trọng): số kênh, projector, partially-cross, đầu vào D, MVG, lịch `λ`.
- Chạy: `sh experiments/run_cakd.sh` (cần đường dẫn ImageNet, nhiều GPU).

### 12.4 Hạn chế / hướng mở

- Bản công khai **không kèm teacher & augmentation tùy biến** ⇒ khó đạt số tuyệt đối như paper.
- Phương pháp hiện cho **phân loại** là chính (dù khái quát được sang detection/segmentation).
- Lịch `λ` và các trọng số (0,2 / 0,05) cần dò theo từng cặp giáo viên–học sinh.

---

*Tài liệu phân tích này dựa trên: bản dịch tiếng Việt đầy đủ của bài báo (`2207.05273v2_vi_full_translation.docx`), bản PDF gốc, `README.md`, sơ đồ `img/framework.png`, và toàn bộ mã nguồn trong `CAKD/`.*


