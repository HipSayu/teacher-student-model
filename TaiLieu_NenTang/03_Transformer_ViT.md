# 03 — Transformer & Vision Transformer (ViT)

Transformer (cụ thể **ViT-B/16**) là **giáo viên** trong CAKD. File này xây từ cơ chế attention đến toàn bộ ViT, kèm **shape đầu vào/đầu ra** và đúng những đầu ra mà CAKD "rút" (`Q_T,K_T,V_T`, `h_T`, class token, logits).

---

## 1. Trực giác: attention là gì?

Hãy tưởng tượng đọc câu "con mèo ngồi trên *nó*". Để hiểu "nó" là gì, ta phải **nhìn lại** "con mèo". **Attention** chính là cơ chế cho mỗi phần tử *tự quyết định nên chú ý vào những phần tử nào khác* và lấy thông tin từ chúng — **toàn cục, ngay lập tức**.

So với CNN (nhìn cục bộ, tăng dần), attention cho **quan hệ xa tùy ý** ngay từ lớp đầu. Đây là thế mạnh CAKD muốn chuyển sang CNN.

---

## 2. Query, Key, Value (Q, K, V)

Ẩn dụ thư viện:
- **Query (truy vấn):** "tôi đang cần gì".
- **Key (khóa):** "tôi mô tả nội dung của tôi thế nào".
- **Value (giá trị):** "nội dung thực sự của tôi".

Mỗi token tạo ra cả ba bằng 3 phép chiếu tuyến tính từ embedding đầu vào `x`:

```
Q = x W_Q,  K = x W_K,  V = x W_V
Input: x = (B, N, D)   →   Q,K,V = (B, N, d)   (d = số chiều mỗi đầu attention)
```

---

## 3. Scaled Dot-Product Attention — trái tim của Transformer

```
Attention(Q,K,V) = softmax( Q K^T / sqrt(d) ) · V
```

**Diễn giải từng bước (kèm shape):**

```
1) Điểm số tương đồng:  S = Q K^T            (B,N,d)·(B,d,N) → (B, N, N)
2) Chia tỉ lệ:          S = S / sqrt(d)       giữ nguyên (B,N,N)  ← tránh số quá lớn làm softmax bão hòa
3) Chuẩn hóa:           A = softmax(S, -1)    (B,N,N)  mỗi hàng = phân phối "chú ý vào ai"
4) Tổng hợp:            O = A · V             (B,N,N)·(B,N,d) → (B, N, d)
```

- `S[i,j]` = mức token i nên chú ý tới token j. `A` = **bản đồ chú ý (attention map)**.
- `O[i]` = tổng có trọng số của các Value → biểu diễn mới của token i đã "thu thập thông tin toàn cục".

```
   token i  ──Q_i──┐
                   ▼
   [Q_i · K_j cho mọi j]  →  softmax  →  trọng số a_ij  →  O_i = Σ_j a_ij V_j
                   ▲
   mọi token j ──K_j,V_j──┘
```

> Liên hệ CAKD (rất quan trọng): **bản đồ `S = QK^T/√d`** (trước softmax) và **Gram `VV^T`** chính là thứ học sinh phải bắt chước trong công thức (4). File `functional.py` được sửa để trả về đúng `[S, VV^T]` này.

---

## 4. Multi-Head Attention (MHA) — nhiều "góc nhìn"

Thay vì 1 attention với `D` chiều, ta chạy **h đầu (head)** song song, mỗi đầu `d = D/h` chiều, rồi ghép lại:

```
head_k = Attention(Q_k, K_k, V_k)        k = 1..h
MHA    = Concat(head_1,...,head_h) · W_O
Input: x=(B,N,D)  →  Output: (B,N,D)     (ViT-B/16: D=768, h=12, d=64)
```

Trực giác: mỗi đầu học một kiểu quan hệ khác nhau (đầu này lo bố cục, đầu kia lo màu sắc…). Đa dạng hóa biểu diễn.

> Liên hệ CAKD: PCA projector của học sinh dùng `heads=16, dim_head=64` (1024 = 16×64). Khi so khớp, code **gộp trung bình các đầu** (`attn.sum(1)/num_heads`) để ra một bản đồ `(B,196,196)`.

---

## 5. Encoder Block — một "tầng" Transformer

Mỗi block xếp: **chuẩn hóa → attention → cộng tắt → chuẩn hóa → MLP → cộng tắt** (kiểu pre-norm):

```
x ─┬─ LN ─ MHA ───(+)─┬─ LN ─ MLP ───(+)── out
   └───────────────┘   └───────────────┘
   (residual 1)        (residual 2)

MLP: Linear(D→4D) → GELU → Linear(4D→D)      (ViT-B/16: 768→3072→768)
Input/Output mỗi block: (B, N, D) → (B, N, D)   (giữ nguyên shape)
```

- **Residual** (giống ResNet) giúp huấn luyện sâu ổn định.
- **LayerNorm** chuẩn hóa theo chiều đặc trưng từng token.
- **MLP** trộn thông tin trong từng token (theo chiều đặc trưng).

> Liên hệ CAKD: code lấy attention ở **2 block cuối** (`num_layers-2` và `num_layers-1`), mỗi block trả `[QK map, VV map]` → tổng 4 ma trận.

---

## 6. Vision Transformer (ViT) — áp Transformer cho ảnh

ViT (Dosovitskiy và cộng sự, 2021) biến ảnh thành "câu gồm các patch":

```
1) Chia patch:      ảnh (B,3,224,224) → 196 mảnh 16×16×3       (224/16=14, 14×14=196)
2) Patch embedding: mỗi mảnh (16·16·3=768 số) ── Linear ──► véc-tơ 768   → (B,196,768)
3) Class token:     ghép thêm 1 token học được vào đầu chuỗi   → (B,197,768)
4) Positional emb:  cộng véc-tơ vị trí (197×768) để biết thứ tự → (B,197,768)
5) Encoder × 12:    qua 12 block (giữ (B,197,768))
6) Lấy class token: x[:,0] (B,768) ── MLP Head ──► logits (B,1000)
```

```
                ┌── patch 1 ──► emb ─┐
 ảnh 224×224 ──►│   ...              ├─► [CLS, p1, p2, ..., p196] + pos ─► [Encoder×12] ─► LN
                └── patch 196 ─► emb ┘                                          │
                                                                  CLS token ────┴──► MLP Head ─► lớp
```

### Thông số ViT-B/16 ("Base", patch 16)

| Thành phần | Giá trị |
|---|---|
| Số patch `N` | 196 (+1 class token = 197) |
| Chiều ẩn `D` | 768 |
| Số block `L` | 12 |
| Số đầu `h` | 12 (mỗi đầu `d=64`) |
| MLP ẩn | 3072 (= 4×768) |
| Tham số | ~86 triệu |

> **`3hw` = 768?** Bài báo gọi đặc trưng giáo viên `h_T ∈ R^{N×3hw}`. Với patch 16×16×3: `3hw = 3·16·16 = 768` — đúng bằng `D`. Đó là lý do GL projector phải đưa kênh CNN (1024) về **768**.

---

## 7. Chính xác những gì CAKD "rút" từ giáo viên

`vision_transformer.py` được sửa để `forward` trả về **4 thứ**:

```python
return x, attn_weights, cls_token, feats
#      │   │             │          └ feats   = h_T : (B,196,768)   ← khớp GL (công thức 6)
#      │   │             └ cls_token        : (B,768)               ← khớp cls_proj học sinh
#      │   └ attn_weights = [QK,VV (block kế cuối), QK,VV (block cuối)]  ← khớp PCA (công thức 4)
#      └ logits          : (B,1000)                                 ← khớp logits (gl_loss)
```

Trong đó `attn_weights[2] = QK_blockcuối`, `attn_weights[3] = VV_blockcuối`; khi dùng sẽ cắt class token `[:,1:,1:]` để còn `(B,196,196)` khớp số patch học sinh.

---

## 8. Đầu vào / Đầu ra tổng kết của giáo viên ViT

```
ĐẦU VÀO :  ảnh (B, 3, 224, 224)
ĐẦU RA  :  - logits        (B, 1000)
           - attn_weights  4 × (B, 197, 197)   → QK & VV của 2 block cuối
           - cls_token     (B, 768)
           - feats h_T     (B, 196, 768)
(Giáo viên ĐÓNG BĂNG: eval(), không cập nhật gradient — chỉ làm "nguồn tri thức")
```

---

## 9. CNN vs Transformer — bảng đối chiếu (lý do KD xuyên kiến trúc khó)

| Tiêu chí | CNN (ResNet, học sinh) | Transformer (ViT, giáo viên) |
|---|---|---|
| Đơn vị xử lý | feature map `(C,H,W)` | chuỗi token `(N,D)` |
| Phạm vi nhìn | cục bộ, tăng dần | toàn cục ngay lập tức |
| Trộn thông tin | tích chập (lân cận) | attention (mọi cặp) |
| Chuẩn hóa | BatchNorm | LayerNorm |
| Thiên kiến | mạnh (local) → cần ít dữ liệu | yếu → cần nhiều dữ liệu/pretrain |
| Triển khai phần cứng | rất thân thiện | nặng hơn |

→ Khác biệt này = "khoảng cách kiến trúc" mà PCA + GL của CAKD phải bắc cầu.

→ Tiếp theo: `04_GAN.md` để hiểu phần **huấn luyện đối kháng đa khung nhìn**.

---

## Tài liệu tham khảo

- Vaswani và cộng sự (2017) — *Attention Is All You Need* (Transformer gốc).
- Dosovitskiy và cộng sự (2021) — *An Image is Worth 16×16 Words* (ViT).
- Liu và cộng sự (2021) — *Swin Transformer* (một giáo viên khác trong CAKD).
- Ba, Kiros, Hinton (2016) — Layer Normalization.
