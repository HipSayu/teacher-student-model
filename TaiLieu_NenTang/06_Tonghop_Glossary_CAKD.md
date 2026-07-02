# 06 — Tổng hợp: trace input→output toàn khung CAKD + Glossary

File này **ghép mọi mảnh** (01–05) lại: đi theo **một batch ảnh** từ đầu vào tới mọi loss, ghi rõ **shape** từng bước; kèm bảng thuật ngữ Anh–Việt và tài liệu tham khảo.

---

## 1. Trace đầy đủ một lượt huấn luyện (forward) — kèm shape

Giả định: `B` ảnh `224×224`, giáo viên **ViT-B/16**, học sinh **ResNet50**.

```
══════════════════ ĐẦU VÀO ══════════════════
image  : (B, 3, 224, 224)          target : (B,)
(qua MVG/augmentation: cùng shape, đã biến đổi "đa khung nhìn")

══════════════ NHÁNH GIÁO VIÊN (ViT, đóng băng, eval) ══════════════
image → patchify+embed → +cls +pos → [Encoder×12] →
   tea_logits        : (B, 1000)
   tea_attn_weights  : 4 × (B,197,197)   # [QK,VV blok kế-cuối ; QK,VV blok cuối]
   tea_token (cls)   : (B, 768)
   tea_feat  (h_T)   : (B, 196, 768)

══════════════ NHÁNH HỌC SINH (ResNet50, huấn luyện) ══════════════
image → conv1→bn→relu→maxpool → layer1 → layer2 → layer3
   x_3 (h_S)         : (B, 1024, 14, 14)
   tmp = reshape+permute(x_3)            : (B, 196, 1024)   # "token hóa" đặc trưng CNN
   ├─ pca_proj(tmp) → attn_qk, attn_vv   : mỗi cái (B, 196, 196)   # sau khi gộp 16 đầu
   └─ gl_proj(tmp)  → vit_feat (h'_S)    : (B, 196, 768)
   x_3 → layer4 → avgpool → flatten      : (B, 2048)
   ├─ fc       → output (logits)         : (B, 1000)
   └─ cls_proj → proj_token              : (B, 768)

══════════════ DISCRIMINATOR (PatchGAN) ══════════════
input_d_real = tea_attn_weights[2][:,1:,1:]  : (B,1,196,196)  # attention giáo viên (thật)
input_d_fake = attn_qk (học sinh)            : (B,1,196,196)  # attention học sinh (giả)
pred_real, pred_fake = D(...)                : (B, 1)
```

### Các loss được tính từ những tensor trên

```
cls_loss = CrossEntropy(output, target)                                   # (B,1000)+(B,) → 1 số

pca_loss = 0.2 ·MSE(attn_qk , tea_QK_cuối[:,1:,1:])                        # khớp bản đồ chú ý   ── (4) số hạng 1
         + 0.05·MSE(attn_vv , tea_VV_cuối[:,1:,1:])                        # khớp Gram Value     ── (4) số hạng 2

gl_loss  = MSE(output, tea_logits)                                        # khớp logits (thêm)
         + MSE(proj_token, tea_token)                                     # khớp class-token (thêm)
         + 0.05·MSE(vit_feat, tea_feat)                                   # khớp đặc trưng patch ── (6)

gan_loss = 0.5·( BCE(pred_real, 1) + BCE(pred_fake, 0) )                  # cập nhật D           ── (8)

λ        = min(max(epoch−25,0)/50, 0.2)                                   # lịch warm-up (0→0.2)
loss     = cls_loss + λ·( pca_loss + gl_loss
                          + 0.05·BCE(pred_real,1) + BCE(pred_fake,1) )    # cập nhật học sinh    ── (10)+(9)
```

---

## 2. Sơ đồ "ai khớp với ai"

```
   HỌC SINH (CNN)                         GIÁO VIÊN (Transformer)
   ───────────────                        ───────────────────────
   attn_qk  (196×196) ───MSE 0.2────────► QK map block cuối (196×196)     │ PCA  (4)
   attn_vv  (196×196) ───MSE 0.05───────► VV map block cuối (196×196)     │
   vit_feat (196×768) ───MSE 0.05───────► h_T = feats      (196×768)      │ GL   (6)
   proj_token (768)   ───MSE──────────► cls_token         (768)          │ (thêm)
   output (1000)      ───MSE──────────► tea_logits        (1000)         │ (thêm)
   attn_qk (giả) ──► D ◄── QK giáo viên (thật)  → đối kháng              │ GAN  (8)(9)
   output ──CrossEntropy── nhãn thật                                     │ giám sát
```

---

## 3. Hai vòng cập nhật luân phiên (mỗi batch)

```
Bước 1 — cập nhật DISCRIMINATOR:
   d_optimizer.zero_grad()
   gan_loss.backward(retain_graph=True)   # ∇ của (8)
   d_optimizer.step()                     # lr = 0.01 × lr_học_sinh (D học chậm)

Bước 2 — cập nhật HỌC SINH (Θ_S + PCA + GL):
   optimizer.zero_grad()
   loss.backward()                        # ∇ của (10)
   optimizer.step()
```

`λ = 0` trong 25 epoch đầu ⇒ học sinh chỉ học phân loại cho "đứng vững", rồi mới bật dần chưng cất tới `0.2`.

---

## 4. Vòng đời mô hình: huấn luyện → suy luận

```
HUẤN LUYỆN:  Θ_S(CNN) + Proj1(PCA) + Proj2(GL) + D + Θ_T(ViT đóng băng)
                                  │  hội tụ
SUY LUẬN  :  CHỈ Θ_S(CNN):  image (B,3,224,224) → logits (B,1000)
             (bỏ PCA, GL, D, ViT → không thêm chi phí)
```

---

## 5. Glossary — thuật ngữ Anh ↔ Việt (tra nhanh)

| Tiếng Anh | Tiếng Việt | Ghi chú nhanh |
|---|---|---|
| Knowledge Distillation (KD) | Chưng cất tri thức | thầy dạy trò |
| Teacher / Student | Giáo viên / Học sinh | ViT / ResNet |
| Cross-Architecture | Xuyên kiến trúc | Transformer→CNN |
| Convolution | Tích chập | kernel trượt trên ảnh |
| Feature map | Bản đồ đặc trưng | đầu ra conv `(C,H,W)` |
| Channel | Kênh | chiều `C` |
| Stride / Padding | Bước trượt / Đệm | điều khiển kích thước |
| Receptive field | Vùng tiếp nhận | vùng ảnh một nơ-ron "thấy" |
| Residual / Skip connection | Khối còn lại / Kết nối tắt | `out = F(x)+x` |
| Bottleneck | Khối nút cổ chai | 1×1→3×3→1×1 |
| Pooling / GAP | Gộp / Gộp trung bình toàn cục | thu nhỏ không gian |
| Attention | Cơ chế chú ý | mỗi token nhìn mọi token |
| Query / Key / Value | Truy vấn / Khóa / Giá trị | Q,K,V |
| Self-attention | Tự chú ý | Q,K,V cùng từ một chuỗi |
| Multi-head | Đa đầu | nhiều attention song song |
| Attention map | Bản đồ chú ý | `softmax(QK^T/√d)` |
| Patch | Mảnh ảnh | ô 16×16 |
| Patch embedding | Nhúng mảnh | mảnh → véc-tơ |
| Class token | Token lớp | gom thông tin để phân loại |
| Positional embedding | Nhúng vị trí | thêm thông tin thứ tự |
| Encoder block | Khối mã hóa | LN+MHA+MLP+residual |
| MLP head | Đầu MLP | lớp phân loại cuối |
| LayerNorm / BatchNorm | Chuẩn hóa lớp / theo lô | LN cho ViT, BN cho CNN |
| Logits | Điểm số trước softmax | đầu ra phân loại thô |
| Softmax | Hàm softmax | điểm số → xác suất |
| Cross-Entropy | Entropy chéo | loss phân loại |
| MSE | Sai số bình phương TB | loss khớp đặc trưng |
| Cosine similarity | Độ tương đồng cosin | đo "khả năng chuyển giao" |
| Gram matrix | Ma trận Gram | `XX^T`, quan hệ tương hỗ |
| GAN | Mạng đối kháng sinh | G vs D |
| Generator / Discriminator | Bộ sinh / Bộ phân biệt | học sinh / D trong CAKD |
| Adversarial loss | Mất mát đối kháng | làm D nhầm |
| PatchGAN | GAN theo mảng | D chấm điểm theo vùng |
| Mode collapse | Sụp chế độ | G mất đa dạng |
| Dropout | Bỏ học ngẫu nhiên | chống overfit |
| SGD / Momentum | Hạ dốc ngẫu nhiên / Quán tính | tối ưu |
| Weight decay | Suy giảm trọng số | chính quy hóa |
| EMA | Trung bình trượt mũ | trọng số "mượt" |
| Warm-up | Khởi động (tăng dần) | `λ` tăng dần |
| Inference | Suy luận | giai đoạn dùng mô hình |
| FLOPs | Số phép tính dấu phẩy động | đo chi phí tính toán |
| PCA projector | Bộ chiếu chú ý giao chéo từng phần | (1)–(4) |
| GL projector | Bộ chiếu tuyến tính theo nhóm | (5)–(6) |
| MVG | Bộ sinh đa khung nhìn | (7) |

---

## 6. Bảng số chiều "phải nhớ" (ViT-B/16 ↔ ResNet50, ảnh 224)

| Đại lượng | Giá trị | Vì sao |
|---|---|---|
| Số patch / token `N` | 196 (+1 cls = 197) | 224/16=14, 14² =196 |
| Chiều embedding ViT `D` = `3hw` | 768 | 3·16·16 |
| Số đầu attention ViT `h` (d=64) | 12 | 768/64 |
| Số block ViT `L` | 12 | "Base" |
| Kênh ResNet sau layer3 (`h_S`) | 1024, lưới 14×14 | nơi gắn projector |
| Kênh ResNet sau layer4 | 2048, lưới 7×7 | trước GAP |
| GL projector | 1024 → 768, **16 lớp FC** | nhóm 4×4 patch chung 1 FC |
| PCA projector | 16 đầu × 64 | 16·64 = 1024 |

---

## 7. Tự kiểm tra hiểu bài (câu hỏi ôn)

1. Vì sao không thể MSE *trực tiếp* `(B,1024,14,14)` của CNN với `(B,196,768)` của ViT? (Gợi ý: shape & ý nghĩa khác.)
2. PCA chuyển CNN vào *không gian nào*? GL chuyển vào *không gian nào*? Mỗi cái dạy học sinh điều gì?
3. Trong công thức (4), số hạng `VV^T` đóng vai trò gì ngoài bản đồ `QK^T`?
4. Vì sao `λ` để = 0 trong 25 epoch đầu?
5. Khi suy luận, những thành phần nào bị bỏ? Lợi ích?
6. Khác biệt giữa paper và code ở đầu vào Discriminator là gì?

> Đáp án nằm rải trong các file 01–05 và `../Phan_tich_chi_tiet_CAKD.md`.

---

## 8. Tài liệu tham khảo gốc (đọc thêm)

**Nền tảng:**
- He và cộng sự (2016) — ResNet. · Vaswani và cộng sự (2017) — Transformer. · Dosovitskiy và cộng sự (2021) — ViT. · Goodfellow và cộng sự (2014) — GAN. · Hinton và cộng sự (2015) — KD.

**Liên quan trực tiếp tới CAKD:**
- Liu và cộng sự (2022) — *Cross-Architecture Knowledge Distillation*, ACCV (bài báo gốc, arXiv:2207.05273).
- Romero (FitNet), Zagoruyko (AT), Park (RKD), Tian (CRD), Touvron (DeiT), Isola (PatchGAN) — các phương pháp được so sánh/sử dụng.

---

*Hết bộ tài liệu nền tảng. Quay lại `00_Mucluc_Tongquan.md` để xem sơ đồ tổng thể, hoặc `../Phan_tich_chi_tiet_CAKD.md` để đọc phân tích công thức–code chi tiết.*
