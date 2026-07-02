# Bộ tài liệu nền tảng cho bài báo CAKD — Mục lục & Tổng quan

Bộ tài liệu này bổ sung **kiến thức nền** để hiểu thật sâu bài báo *Cross-Architecture Knowledge Distillation* (CAKD, ACCV 2022). Mỗi file đi **từ cơ bản đến nâng cao**, có **sơ đồ ASCII**, và đặc biệt luôn ghi rõ **đầu vào → đầu ra (shape tensor)** ở từng bước.

> Đọc kèm file phân tích bài báo: `../Phan_tich_chi_tiet_CAKD.md`.

---

## Thứ tự đọc đề xuấtV

| # | File | Nội dung | Vì sao cần cho CAKD |
|---|---|---|---|
| 01 | `01_Toan_va_Khainiem_Nentang.md` | Tensor, nhân ma trận, softmax, cross-entropy, MSE, cosine, Gram, gradient/SGD, backprop, BatchNorm/LayerNorm, dropout, ReLU/GELU, EMA, embedding | Mọi công thức (1)–(10) đều dựng từ các viên gạch này |
| 02 | `02_CNN.md` | Tích chập, kernel/stride/padding, feature map, pooling, receptive field, ResNet/residual/bottleneck, ResNet50 từng stage | Hiểu **học sinh** (ResNet50) và đặc trưng `h_S` |
| 03 | `03_Transformer_ViT.md` | Attention Q/K/V, scaled dot-product, multi-head, positional encoding, patch embedding, class token, encoder block, MLP head, ViT-B/16 | Hiểu **giáo viên** (ViT) và `Q_T,K_T,V_T`, `h_T` |
| 04 | `04_GAN.md` | Generator/Discriminator, minimax, BCE, non-saturating loss, PatchGAN, mode collapse | Hiểu **MVG + Discriminator** và loss (7)–(9) |
| 05 | `05_Knowledge_Distillation.md` | Teacher–student, soft target, temperature, logit/feature/attention KD, bức tranh SOTA | Hiểu **bản chất KD** và vị trí của CAKD |
| 06 | `06_Tonghop_Glossary_CAKD.md` | Trace input→output toàn khung CAKD, bảng thuật ngữ Anh–Việt, tài liệu tham khảo | Ghép tất cả lại thành một bức tranh |

---

## Bản đồ khái niệm: cái gì dùng ở đâu trong CAKD

```
                          ┌──────────────── KIẾN THỨC NỀN ────────────────┐
                          │  Toán (01): softmax, MSE, cosine, Gram, SGD    │
                          └───────┬───────────────┬───────────────┬───────┘
                                  │               │               │
                     ┌────────────▼───┐   ┌───────▼────────┐   ┌──▼─────────────┐
                     │  CNN (02)      │   │ Transformer(03)│   │   GAN (04)     │
                     │  = HỌC SINH    │   │ = GIÁO VIÊN    │   │ = MVG + D      │
                     └────────────┬───┘   └───────┬────────┘   └──┬─────────────┘
                                  │               │               │
                                  └───────┬───────┴───────┬───────┘
                                          ▼               ▼
                                  ┌─────────────────────────────┐
                                  │  Knowledge Distillation (05) │
                                  │  = khung teacher–student     │
                                  └──────────────┬──────────────┘
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │  CAKD (bài báo) = 02+03+04+05 │
                                  │  PCA (1-4) · GL (5-6) ·       │
                                  │  MVG/GAN (7-9) · Tổng (10)    │
                                  └─────────────────────────────┘
```

---

## Quy ước ký hiệu shape dùng xuyên suốt

- `B` = batch size (số ảnh mỗi lô). Ví dụ mặc định: 224×224, ResNet50 ↔ ViT-B/16.
- `C` = số kênh (channels); `H, W` = chiều cao/rộng không gian.
- `N` = số token/patch (ViT-B/16 trên ảnh 224 → **196** patch + 1 class token = **197**).
- `D` hoặc `E` = số chiều embedding (ViT-B/16: **768**); `d` = số chiều mỗi đầu attention (**64**).
- Ký hiệu shape viết dạng `(B, C, H, W)` cho ảnh/CNN và `(B, N, D)` cho chuỗi token/Transformer.

> Ví dụ nhanh để ghi nhớ: ảnh `(B,3,224,224)` → ResNet50 sau `layer3` cho `(B,1024,14,14)`; ViT-B/16 cho chuỗi `(B,197,768)`.
