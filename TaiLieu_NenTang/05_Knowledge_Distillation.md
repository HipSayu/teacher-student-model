# 05 — Knowledge Distillation (Chưng cất tri thức, KD)

Đây là **khung tổng** mà CAKD thuộc về. File này giải thích KD từ ý tưởng gốc (Hinton) đến các nhánh hiện đại, kèm input/output, rồi định vị chính xác CAKD trên bản đồ KD.

---

## 1. Trực giác: "thầy giỏi dạy trò nhỏ"

Một mô hình lớn (**giáo viên/teacher**) mạnh nhưng nặng. Ta muốn một mô hình nhỏ (**học sinh/student**) nhẹ mà vẫn giỏi. KD = huấn luyện học sinh *bắt chước* giáo viên, không chỉ học từ nhãn.

Vì sao hiệu quả? Vì giáo viên cung cấp **tín hiệu giàu hơn nhãn cứng**. Nhãn "mèo" chỉ nói đúng/sai; nhưng phân phối mềm của giáo viên nói "90% mèo, 8% chó, 2% hổ" — *hé lộ cấu trúc giữa các lớp* ("dark knowledge").

```
            nhãn cứng:  [0, 1, 0, 0]            (chỉ biết "mèo")
  giáo viên mềm:        [0.02, 0.90, 0.06, 0.02] (mèo, nhưng hơi giống chó/hổ)
                         ▲ thông tin "ẩn" này giúp học sinh học nhanh & khái quát tốt hơn
```

---

## 2. KD kinh điển trên logits (Hinton và cộng sự, 2015)

### Nhiệt độ (temperature) `T` làm "mềm" phân phối

```
p_i(T) = softmax(z_i / T) = e^{z_i/T} / Σ_j e^{z_j/T}
T = 1  → phân phối gốc;   T > 1 → mềm hơn (lộ rõ tương quan lớp)https://meet.google.com/yoo-heoi-pkv
```

### Hàm mất mát KD

```
L = α · CE(p_student(T=1), nhãn_thật)                     ← học từ nhãn
  + (1−α) · T² · KL( p_student(T) || p_teacher(T) )       ← bắt chước giáo viên
```

- `KL` = phân kỳ Kullback–Leibler (đo khác biệt 2 phân phối). Nhân `T²` để cân bằng độ lớn gradient.

```
Input : logits_student (B,C), logits_teacher (B,C), nhãn (B,)
Output: 1 số (loss)
```

> Liên hệ CAKD: trong code, KD logits được hiện thực **đơn giản bằng MSE** `mse_criterion(output, tea_logits)` (một số hạng của `gl_loss`), thay vì KL+temperature.

---

## 3. KD trên đặc trưng trung gian (feature / hint KD)

Thay vì chỉ khớp đầu ra, ép **đặc trưng lớp giữa** của học sinh giống giáo viên. Vấn đề: số chiều thường lệch nhau → cần một **bộ chiếu/regressor** để căn chỉnh.

```
L_hint = || f_teacher − Proj(f_student) ||²        (MSE sau khi căn chỉnh chiều)
```

> Đây chính là họ phương pháp mà CAKD thuộc về — nhưng nâng cấp cho **xuyên kiến trúc** bằng hai bộ chiếu chuyên dụng (PCA, GL).

---

## 4. Bản đồ các nhánh KD (để hiểu "14 SOTA" bài báo so sánh)

| Nhóm | Phương pháp | "Tri thức" được chưng cất |
|---|---|---|
| **Logits** | Hinton (Logits) | phân phối mềm đầu ra |
| **Feature/Hint** | FitNet | đặc trưng lớp giữa (qua regressor) |
| | OFD, AB | đặc trưng + biên kích hoạt |
| **Attention** | AT | bản đồ chú ý gộp-kênh |
| **Quan hệ (relational)** | RKD, IRG | quan hệ *giữa các mẫu* (khoảng cách/góc) |
| | CRD | tương phản (contrastive) giữa teacher–student |
| **Review/đa tầng** | ReviewKD | nối nhiều tầng để chưng cất |
| **Cho Transformer** | DeiT | token chưng cất (hard label từ teacher) |
| | MINILM, IR | tự chú ý / biểu diễn bên trong Transformer |

**Khoảng trống mà CAKD lấp:** gần như tất cả ở trên **giả định giáo viên–học sinh CÙNG (hoặc gần) kiến trúc**. Khi giáo viên là **Transformer** còn học sinh là **CNN**, chúng hoặc không áp dụng được, hoặc kém hiệu quả (vì đặc trưng khác định dạng/ý nghĩa).

```
        Cùng kiến trúc (CNN→CNN, T→T)          Xuyên kiến trúc (T→CNN)
        cosine ≈ 0.6–0.7  ✅ dễ chưng cất       cosine < 0.55  ❌ khó → CAKD ra đời
```

---

## 5. CAKD đứng ở đâu? (định vị)

CAKD = **feature/attention KD** được thiết kế lại cho **xuyên kiến trúc Transformer→CNN**, cộng thêm **học đối kháng**:

```
            ┌─────────────────────── CAKD ───────────────────────┐
   Logits KD│  (gl_loss có MSE logits)                            │
  Feature KD│  GL projector  → khớp đặc trưng patch  (5)-(6)       │
Attention KD│  PCA projector → khớp bản đồ chú ý     (1)-(4)       │
        GAN │  MVG + Discriminator → bền vững         (7)-(9)       │
            └─────────────────────────────────────────────────────┘
                       Tổng: L_total (10), suy luận chỉ giữ CNN
```

Điểm khác biệt cốt lõi so với KD truyền thống: **không bắt chước trực tiếp**, mà **chiếu đặc trưng CNN vào đúng "không gian" của Transformer** (không gian chú ý & không gian đặc trưng) *rồi mới* so khớp. Nhờ vậy cosine (khả năng chuyển giao) tăng vượt cả mức cùng kiến trúc.

---

## 6. Vì sao chỉ giữ CNN khi suy luận lại quan trọng

```
HUẤN LUYỆN:   [CNN học sinh] + PCA + GL + Discriminator + (Transformer giáo viên đóng băng)
                                   │ tất cả chỉ để "dạy"
SUY LUẬN:     [CNN học sinh]  ◄── bỏ hết phần phụ trợ
              → nhẹ, nhanh, thân thiện phần cứng (CUDA/TensorRT/NCNN), nhưng đã "thấm" tri thức Transformer
```

Đây là triết lý "được người khổng lồ dạy, nhưng vẫn nhỏ gọn".

---

## 7. Đầu vào / Đầu ra tổng kết của khung KD (CAKD)

```
ĐẦU VÀO huấn luyện:
  - ảnh + nhãn thật
  - giáo viên ViT (đã pretrain, ĐÓNG BĂNG)
ĐẦU RA huấn luyện:
  - CNN học sinh đã "chưng cất" (kèm projector/discriminator — sẽ bỏ đi)
ĐẦU VÀO suy luận:  ảnh (B,3,224,224)
ĐẦU RA suy luận :  logits (B, num_classes)   — CHỈ dùng CNN học sinh
```

→ Tiếp theo: `06_Tonghop_Glossary_CAKD.md` — trace toàn bộ input→output của CAKD + glossary + tham khảo.

---

## Tài liệu tham khảo

- Hinton, Vinyals, Dean (2015) — *Distilling the Knowledge in a Neural Network*.
- Romero và cộng sự (2015) — *FitNets* (hint/feature KD).
- Zagoruyko, Komodakis (2017) — *Attention Transfer* (AT).
- Park và cộng sự (2019) — *Relational KD* (RKD).
- Tian, Krishnan, Isola (2020) — *Contrastive Representation Distillation* (CRD).
- Touvron và cộng sự (2021) — *DeiT* (KD cho Transformer).
