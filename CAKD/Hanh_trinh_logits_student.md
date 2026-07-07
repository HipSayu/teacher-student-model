# `z_S` (logits student) — từ ảnh gốc ra logits qua những bước nào?

> Giải thích biến `output` trong bảng ký hiệu của [`FORMULAS.md`](FORMULAS.md):
>
> | Ký hiệu | Biến | Shape | Sinh ra tại | Ghi chú |
> |---|---|---|---|---|
> | $z_S$ | `output` | (B, 1000) | [`resnet.py:659`](cakd_modified_files/resnet.py) `x = self.fc(cnn_token)` | logits phân loại |

`z_S` (= biến `output`) là **vector 1000 điểm số thô (logits)** — mỗi số ứng với 1 lớp
trong 1000 lớp của ImageNet. Nó đi ra từ **nhánh phân loại** của student `ResNet_CAKD`.
Dưới đây là toàn bộ hành trình từ ảnh gốc, kèm shape ở từng bước
(trích [`resnet.py:631-659`](cakd_modified_files/resnet.py)).

---

## Hành trình `image → z_S`

```
image                          (B,   3, 224, 224)   ← ảnh RGB đầu vào
  │
  ├─ conv1  (Conv 7×7, stride 2)   → (B,  64, 112, 112)   [dòng 633]  đổi 3 kênh RGB→64, giảm nửa kích thước
  ├─ bn1 → relu                    → (B,  64, 112, 112)   [634-635]  chuẩn hóa + kích hoạt
  ├─ maxpool (3×3, stride 2)       → (B,  64,  56,  56)   [636]      gộp cực đại, giảm nửa nữa
  │        ↑↑↑ 4 bước trên gọi chung là "stem" (cổ vào)
  │
  ├─ layer1  (3 Bottleneck)        → (B, 256,  56,  56)   [638]      trích đặc trưng, nở kênh ×4
  ├─ layer2  (4 Bottleneck, /2)    → (B, 512,  28,  28)   [639]      giảm kích thước, tăng kênh
  ├─ layer3  (6 Bottleneck, /2)    → (B,1024,  14,  14)   [640]  = x_3  ← GIỮ LẠI (còn dùng cho nhánh distill)
  │
  ├─ layer4  (3 Bottleneck, /2)    → (B,2048,   7,   7)   [655]      tầng CNN sâu nhất
  ├─ avgpool (AdaptiveAvgPool 1×1) → (B,2048,   1,   1)   [657]      gộp trung bình toàn ảnh 7×7 → 1 điểm
  ├─ flatten                       → (B,2048)             [658]  = cnn_token  (vector đặc trưng cuối)
  │
  └─ fc  (Linear 2048 → 1000)      → (B,1000)             [659]  = output = z_S   ★ LOGITS
```

---

## Giải thích các mốc quan trọng

- **B** = batch size (số ảnh xử lý cùng lúc); nó đi xuyên suốt không đổi.
- **Kích thước không gian giảm dần** 224 → 112 → 56 → 28 → 14 → 7, còn **số kênh tăng dần**
  3 → 64 → 256 → 512 → 1024 → 2048. Đây là quy luật chung của CNN:
  *"thu nhỏ không gian, dày thêm đặc trưng"*.
- **`x_3` (dòng 640)** là điểm rẽ nhánh: nhánh phân loại đi tiếp qua `layer4` → `fc`,
  còn **nhánh distill** (`pca_proj`, `gl_proj`) cũng lấy `x_3` để sinh $A^{qk}_S, A^{vv}_S, f_S$.
  Nhưng hai nhánh độc lập — **`z_S` KHÔNG phụ thuộc nhánh distill**, chỉ phụ thuộc chuỗi
  `x_3 → layer4 → avgpool → fc`.
- **`cnn_token` (B, 2048)** là "bản tóm tắt đặc trưng" của cả ảnh sau khi CNN xử lý xong.
  Từ đây rẽ 2 đường:
  - `self.fc(cnn_token)` → **`output` = z_S** (logits, dùng cho `cls_loss` và so với `tea_logits`).
  - `self.cls_proj(cnn_token)` → **`proj_token` = t_S** (chiếu sang 768 chiều để khớp class token teacher).

---

## Vậy `z_S` là gì về mặt ý nghĩa?

`z_S = [s₀, s₁, ..., s₉₉₉]` — 1000 số thực **chưa chuẩn hóa**. Số nào lớn nhất → model đoán
ảnh thuộc lớp đó. Muốn ra xác suất thì đưa qua softmax (nhưng `CrossEntropyLoss` đã gộp sẵn
softmax bên trong nên code không softmax thủ công).

Ví dụ trực giác: nếu `argmax(z_S) = 285` và lớp 285 của ImageNet là *"mèo Ai Cập"*
→ model dự đoán ảnh là mèo Ai Cập.

---

## Ghi chú số kênh (ResNet-50)

Mỗi `layerN` gồm nhiều khối **Bottleneck**, mỗi Bottleneck nở kênh gấp `expansion = 4`:

| Tầng | Số block | Kênh gốc | Kênh ra (×4) | Kích thước |
|---|---|---|---|---|
| layer1 | 3 | 64  | 256  | 56×56 |
| layer2 | 4 | 128 | 512  | 28×28 |
| layer3 | 6 | 256 | 1024 | 14×14 (= `x_3`) |
| layer4 | 3 | 512 | 2048 | 7×7 |

Lớp phân loại cuối: `self.fc = nn.Linear(512 * expansion, num_classes)` = `Linear(2048, 1000)`
([`resnet.py:569`](cakd_modified_files/resnet.py)).

---

*Nguồn: `ResNet_CAKD._forward_impl` trong [`cakd_modified_files/resnet.py`](cakd_modified_files/resnet.py) dòng 631–666.*
