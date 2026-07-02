# 04 — GAN (Generative Adversarial Network) & học đối kháng

CAKD dùng tư tưởng GAN trong **lược đồ huấn luyện bền vững đa khung nhìn**: một **Discriminator** phân biệt đặc trưng giáo viên (thật) vs học sinh (giả), còn học sinh đóng vai **Generator** cố làm Discriminator nhầm. File này giải thích GAN từ gốc, kèm input/output, rồi nối về công thức (7)–(9).

---

## 1. Trực giác: trò chơi "làm giả vs bắt giả"

Hai mạng đối đầu:
- **Generator G (kẻ làm giả):** tạo dữ liệu giả sao cho *giống thật*.
- **Discriminator D (cảnh sát):** phân biệt thật vs giả.

Cả hai cùng mạnh dần: G làm giả ngày càng tinh vi, D bắt giả ngày càng giỏi. Tại điểm cân bằng lý tưởng, đồ giả của G *không phân biệt được* với thật → G đã học được phân phối dữ liệu thật.

```
  nhiễu z ─► [ G ] ─► mẫu giả ─┐
                                ├─► [ D ] ─► xác suất "thật?" ∈ (0,1)
  dữ liệu thật ────────────────┘
```

> Trong CAKD **không sinh ảnh**. "Thật" = đặc trưng/bản đồ chú ý của **giáo viên**; "giả" = của **học sinh**. Mục tiêu: đặc trưng học sinh có *phân phối giống* giáo viên.

---

## 2. Hàm mục tiêu minimax (GAN gốc, Goodfellow 2014)

```
min_G max_D  E_{x~thật}[ log D(x) ]  +  E_{z~nhiễu}[ log(1 − D(G(z))) ]
```

- **D muốn:** `D(thật) → 1`, `D(giả) → 0` ⇒ **tăng** biểu thức.
- **G muốn:** `D(giả) → 1` ⇒ **giảm** biểu thức (làm D nhầm).

Đây là lý do gọi là **đối kháng (adversarial)**: hai mục tiêu ngược nhau, tối ưu **luân phiên**.

---

## 3. Loss cụ thể bằng BCE

`D` xuất ra logit; qua sigmoid thành xác suất. Dùng **Binary Cross-Entropy**:

### Cập nhật Discriminator (nhãn: thật=1, giả=0)

```
L_D = − [ log D(thật) + log(1 − D(giả)) ]
```

### Cập nhật Generator — hai biến thể

```
(a) Saturating  (đúng lý thuyết):  L_G = log(1 − D(giả))        ← gradient yếu lúc đầu
(b) Non-saturating (thực dụng):    L_G = − log D(giả)           ← mạnh hơn, hay dùng
```

> Liên hệ CAKD:
> - **(8) `L_MAD`** = chính là `L_D` ở trên (cập nhật Discriminator).
> - **(9) `L_MVG`** = `log(1 − D(giả))` (dạng saturating trong paper). **Code** lại dùng dạng **non-saturating** `gan_criterion(pred_fake, True)` (ép `D(giả)→thật`) cho ổn định gradient.

---

## 4. PatchGAN — Discriminator "chấm điểm theo mảng"

Thay vì xuất 1 số cho cả ảnh, **PatchGAN** (Isola và cộng sự, 2017) chạy conv để chấm điểm thật/giả cho **từng vùng**, rồi tổng hợp. Ưu điểm: ít tham số, tập trung vào kết cấu cục bộ.

```
Input (B, C, H, W) ─► [Conv s2 → LeakyReLU] × n ─► Conv → bản đồ điểm ─► (GAP) ─► điểm thật/giả
```

> Liên hệ CAKD: `new_utils.NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3)` chính là PatchGAN.
> - **Đầu vào:** bản đồ chú ý **1 kênh** `(B,1,196,196)` (lấy từ attention map giáo viên hoặc học sinh).
> - Qua các conv `kernel=4, stride=2` + `LeakyReLU(0.2)` → cuối cùng `AdaptiveAvgPool2d(1)` → **một điểm số** "thật/giả".
> - `ndf=8` (số bộ lọc nhỏ) ⇒ Discriminator gọn nhẹ.

---

## 5. Vì sao GAN khó huấn luyện?

- **Mode collapse:** G chỉ tạo một kiểu mẫu để "lừa" D ⇒ mất đa dạng.
- **Mất cân bằng:** nếu D quá mạnh, gradient cho G ~ 0 (G không học được).
- **Dao động:** loss không hội tụ êm.

**Mẹo ổn định** thường dùng (và có trong CAKD):
- D học **chậm hơn** G: trong CAKD, `d_optimizer` có **lr = 0,01 × lr** của học sinh.
- **Cập nhật luân phiên** (cập nhật D trước, rồi G).
- **Trọng số nhỏ** cho số hạng đối kháng (CAKD nhân `0,05` cho một nhánh và đặt trong lịch warm-up `λ`).
- Dùng **non-saturating loss**.
- LeakyReLU trong D (tránh nơ-ron chết).

---

## 6. Đối chiếu công thức (7)–(9) ↔ code

```
(7) MVG:  x̃ = Trans(x) với p≥0.5, ngược lại x
          → CODE: thay bằng augmentor PyTorch chuẩn (RandAugment, RandomErasing, Mixup/CutMix)
                  tạo "nhiều khung nhìn" của cùng một ảnh.

(8) L_MAD (cập nhật D):
    L_MAD = (1/m) Σ [ −log D(h_T) − log(1 − D(h'_S)) ]
    → CODE:
      pred_real = discriminator(attn_giáo_viên)      # thật
      pred_fake = discriminator(attn_học_sinh.detach())  # giả (chặn gradient về học sinh)
      gan_loss  = 0.5*( GANLoss(pred_real, True) + GANLoss(pred_fake, False) )
      d_optimizer cập nhật D.

(9) L_MVG (học sinh = Generator):
    L_MVG = (1/m) Σ log(1 − D(h'_S))
    → CODE (non-saturating): ... + gan_criterion(pred_fake, True)   # ép D(học sinh) → "thật"
      nằm trong loss tổng (10) của học sinh.
```

> ⚠️ Khác biệt paper↔code: paper cho D ăn **đặc trưng** `h_T/h'_S`; code cho D ăn **bản đồ chú ý** `(B,1,196,196)`. Cùng tinh thần đối kháng, khác lựa chọn đầu vào.

---

## 7. Đầu vào / Đầu ra tổng kết của phần GAN trong CAKD

```
DISCRIMINATOR D:
  ĐẦU VÀO :  bản đồ chú ý (B, 1, 196, 196)   [của giáo viên HOẶC học sinh]
  ĐẦU RA  :  điểm thật/giả (B, 1)            [logit, qua BCEWithLogits]

HỌC SINH (vai Generator):
  Mục tiêu: làm D chấm bản đồ chú ý của mình là "thật"
  → kéo phân phối đặc trưng học sinh về gần giáo viên, tăng độ bền vững với nhiễu.
```

→ Tiếp theo: `05_Knowledge_Distillation.md` để đặt mọi thứ vào khung teacher–student.

---

## Tài liệu tham khảo

- Goodfellow và cộng sự (2014) — *Generative Adversarial Networks*.
- Radford, Metz, Chintala (2016) — DCGAN (GAN dùng conv).
- Isola và cộng sự (2017) — *pix2pix* (PatchGAN).
- Mao và cộng sự (2017) — LSGAN (một biến thể loss có trong `GANLoss`).
