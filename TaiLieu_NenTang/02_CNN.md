# 02 — CNN (Mạng nơ-ron tích chập), đến ResNet

CNN là **học sinh** trong CAKD (ResNet50). File này giải thích CNN từ gốc, kèm **shape đầu vào/đầu ra** ở từng bước, và chỉ ra chính xác nơi CAKD "rút" đặc trưng `h_S`.

---

## 1. Trực giác: vì sao cần tích chập?

Một bức ảnh 224×224×3 có ~150.000 số. Nếu nối thẳng vào một lớp FC khổng lồ thì:
- quá nhiều tham số (overfit, tốn bộ nhớ),
- mất cấu trúc không gian (điểm ảnh kề nhau có liên hệ).

CNN giải quyết bằng **3 ý tưởng**:
1. **Cục bộ (local):** mỗi nơ-ron chỉ nhìn một vùng nhỏ (ví dụ 3×3).
2. **Chia sẻ trọng số (weight sharing):** cùng một bộ lọc (kernel) trượt khắp ảnh → ít tham số, phát hiện đặc trưng ở mọi vị trí.
3. **Phân cấp (hierarchy):** lớp nông học cạnh/góc; lớp sâu học bộ phận → vật thể.

> Đây chính là **"local inductive bias"** — thế mạnh của CNN, nhưng cũng là lý do CNN **yếu về quan hệ toàn cục** so với Transformer. CAKD bơm phần "toàn cục" còn thiếu này từ ViT sang.

---

## 2. Phép tích chập (convolution)

Một **kernel** (bộ lọc) `(k×k)` trượt trên ảnh, tại mỗi vị trí tính tích vô hướng giữa kernel và vùng ảnh dưới nó → một số. Trượt khắp ảnh → một **feature map** (bản đồ đặc trưng).

```
Vùng ảnh 3×3        kernel 3×3            tích chập tại 1 vị trí
┌─────────┐        ┌─────────┐
│ a b c   │        │ w1 w2 w3│   →  out = a·w1 + b·w2 + c·w3
│ d e f   │   ⊙    │ w4 w5 w6│        + d·w4 + e·w5 + f·w6
│ g h i   │        │ w7 w8 w9│        + g·w7 + h·w8 + i·w9  (+ bias)
└─────────┘        └─────────┘
```

### Tham số then chốt & ảnh hưởng tới shape

- **Kernel size `k`**: vùng nhìn (3×3, 7×7…).
- **Stride `s`**: bước trượt. `s=2` → giảm một nửa kích thước không gian.
- **Padding `p`**: đệm 0 quanh viền để kiểm soát kích thước đầu ra.
- **In-channels `C_in` → Out-channels `C_out`**: mỗi kernel cho 1 kênh đầu ra; có `C_out` kernel.

**Công thức kích thước đầu ra:**

```
H_out = floor( (H_in + 2p − k) / s ) + 1     (tương tự cho W)
Input:  (B, C_in, H_in, W_in)
Kernel: (C_out, C_in, k, k)
Output: (B, C_out, H_out, W_out)
```

Ví dụ conv1 của ResNet: `k=7, s=2, p=3` trên `(B,3,224,224)` → `H_out = (224+6−7)/2 + 1 = 112` → `(B,64,112,112)`.

---

## 3. Pooling — thu nhỏ & bất biến dịch chuyển nhẹ

- **Max pooling 2×2 (s=2):** lấy giá trị lớn nhất mỗi ô 2×2 → giảm một nửa H,W.
- **Average pooling:** lấy trung bình.
- **Global Average Pooling (GAP):** trung bình toàn bộ H×W cho mỗi kênh → `(B,C,H,W) → (B,C,1,1)`.

> Liên hệ CAKD: cuối ResNet, `avgpool` (GAP) biến `(B,2048,7,7) → (B,2048)` trước khi vào FC phân loại.

---

## 4. Receptive field (vùng tiếp nhận)

Càng sâu, một nơ-ron càng "nhìn" được vùng ảnh gốc lớn hơn (do chồng nhiều lớp). Nhưng ngay cả ở lớp sâu, vùng nhìn vẫn **cục bộ và tăng dần** — khác hẳn attention của Transformer vốn **toàn cục ngay từ lớp 1**.

```
Lớp 1: nhìn 3×3      Lớp 2: nhìn ~5×5      Lớp sâu: nhìn vùng lớn nhưng vẫn giới hạn
   (CNN tăng dần)                          (Transformer: toàn ảnh ngay lập tức)
```

---

## 5. Khối còn lại (Residual) & ResNet

Mạng quá sâu bị **vanishing gradient** (gradient teo dần) → khó học. **ResNet** (He và cộng sự, 2016) thêm **kết nối tắt (skip connection)**:

```
        x ───────────────┐ (đường tắt, "identity")
        │                 ▼
   [conv→BN→ReLU→...] →  (+) → ReLU → out        out = F(x) + x
```

Trực giác: mạng chỉ cần học **phần dư** `F(x)` (cái cần thêm vào `x`), dễ hơn nhiều so với học cả ánh xạ. Nhờ vậy huấn luyện được mạng rất sâu (50, 101, 152 lớp).

### Bottleneck block (dùng trong ResNet50/101/152)

Ba lớp conv để rẻ về tính toán: **1×1 giảm kênh → 3×3 xử lý → 1×1 tăng kênh ×4**.

```
Input (B, C, H, W)
 ├─ 1×1 conv: C → C/4        (giảm chiều)   → (B, C/4, H, W)
 ├─ 3×3 conv: C/4 → C/4      (trộn không gian, có thể stride 2)
 ├─ 1×1 conv: C/4 → C        (khôi phục, expansion=4)
 └─ + skip(x) → ReLU
```

---

## 6. ResNet50 đầy đủ — bảng shape từng tầng

`B` = batch; ảnh chuẩn 224×224. **expansion = 4** cho Bottleneck.

```
Tầng        Phép toán                              Output shape         Ghi chú
────────────────────────────────────────────────────────────────────────────────────
Input       —                                      (B,   3, 224, 224)
conv1       7×7, 64, stride 2, pad 3               (B,  64, 112, 112)   stride 2
bn1+relu    —                                      (B,  64, 112, 112)
maxpool     3×3, stride 2, pad 1                   (B,  64,  56,  56)   stride 2
layer1      3 × Bottleneck(64→256)                 (B, 256,  56,  56)
layer2      4 × Bottleneck(128→512), stride 2      (B, 512,  28,  28)   stride 2
layer3      6 × Bottleneck(256→1024), stride 2     (B,1024,  14,  14)   stride 2   ◄── CAKD rút ở đây
layer4      3 × Bottleneck(512→2048), stride 2     (B,2048,   7,   7)   stride 2
avgpool     Global Average Pool                    (B,2048,   1,   1)
flatten     —                                      (B,2048)
fc          Linear 2048 → num_classes              (B, num_classes)     logits
```

- **Tổng tham số ResNet50:** ~25,6 triệu. **Tổng stride:** 32 (224/32 = 7).
- Quy luật: qua mỗi `layer`, **không gian H,W giảm một nửa**, **số kênh C tăng gấp đôi** — đặc trưng "thu gọn không gian, giàu ngữ nghĩa" dần.

---

## 7. Chính xác nơi CAKD lấy đặc trưng học sinh `h_S`

Trong `resnet.py` (lớp `ResNet_CAKD`), hai bộ chiếu gắn **ngay sau `layer3`**:

```
x_3 = layer3(x)                  # (B, 1024, 14, 14)  ← đây là h_S
tmp = x_3.reshape(B,1024,-1)     # (B, 1024, 196)     gộp 14×14 = 196
tmp = tmp.permute(0,2,1)         # (B, 196, 1024)     ◄── đổi sang "định dạng token" giống ViT!
        ├─ pca_proj(tmp) → bản đồ chú ý (B,196,196)   (PCA)
        └─ gl_proj(tmp)  → đặc trưng (B,196,768)      (GL, khớp số chiều ViT)
x   = layer4(x_3) → avgpool → fc # nhánh phân loại vẫn chạy bình thường
```

> **Mấu chốt sư phạm:** lưới không gian `14×14 = 196` của CNN được "ép" thành **196 token**, mỗi token 1024 chiều — *cố tình* trùng số 196 với số patch của ViT-B/16. Đó là cây cầu đầu tiên nối hai kiến trúc.
>
> ⚠️ Paper viết `h_S ∈ R^{256×196}` (256 kênh) còn **code thực tế là 1024 kênh** (sau `layer3`). Khác biệt cấu hình, đã lưu ý trong file phân tích bài báo (Mục 10).

---

## 8. Đầu vào / Đầu ra tổng kết của CNN học sinh

```
ĐẦU VÀO :  ảnh (B, 3, 224, 224)
ĐẦU RA  :  - logits (B, num_classes)        → phân loại / cls_loss
           - h_S    (B, 196, 1024)          → nguyên liệu cho PCA & GL
           - (sau GL)  (B, 196, 768)        → khớp với đặc trưng ViT
           - cls_proj  (B, 768)             → khớp class-token ViT
```

→ Tiếp theo: `03_Transformer_ViT.md` để hiểu **giáo viên** tạo ra `Q_T,K_T,V_T` và `h_T` ra sao.

---

## Tài liệu tham khảo

- LeCun và cộng sự (1998) — LeNet, tích chập cho nhận dạng chữ.
- Krizhevsky, Sutskever, Hinton (2012) — AlexNet.
- He, Zhang, Ren, Sun (2016) — *Deep Residual Learning* (ResNet).
- Sandler và cộng sự (2018) — MobileNetV2 (một học sinh khác trong CAKD).
