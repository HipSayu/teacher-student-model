# Chặng 5 — LUỒNG HOẠT ĐỘNG & LUỒNG TRAIN CỦA CAKD (giải thích cặn kẽ + sơ đồ)

> File này tập trung **VẼ luồng** cho `CAKD/dist_train_cakd.py`. Nếu chặng 1–4 giải thích *code viết gì*, thì chặng này giải thích *dữ liệu chạy như thế nào* — từ tấm ảnh đầu vào tới lúc trọng số được cập nhật.
>
> Đọc kèm 3 file đã comment: `new_utils.py`, `cakd_modified_files/resnet.py`, `cakd_modified_files/vision_transformer.py`.

---

## Mục lục
1. [CAKD là gì? Bài toán gốc](#1)
2. [Ba nhân vật: Student, Teacher, Discriminator](#2)
3. [Bức tranh toàn cảnh 4 file](#3)
4. [Kiến trúc tổng thể (sơ đồ lớn)](#4)
5. [Luồng dữ liệu 1 batch — forward](#5)
6. [4 hàm loss — giải thích từng cái](#6)
7. [Vòng đấu GAN: 2 optimizer chạy xen kẽ](#7)
8. [Cơ chế "khởi động chậm" λ(epoch)](#8)
9. [Luồng train toàn cục (vòng epoch)](#9)
10. [Bảng shape tensor qua từng bước](#10)
11. [Bảng đối chiếu Teacher ↔ Student](#11)
12. [Những chỗ dễ nhầm (FAQ)](#12)

---

<a name="1"></a>
## 1. CAKD là gì? Bài toán gốc

**CAKD = Cross-Architecture Knowledge Distillation** (Chưng cất tri thức xuyên kiến trúc).

**Knowledge Distillation (KD)** = "thầy dạy trò": lấy một model **to, giỏi, chậm** (teacher) để dạy cho một model **nhỏ, nhanh** (student) giỏi theo. Student học bằng cách bắt chước đầu ra của teacher, chứ không chỉ học từ nhãn.

**"Cross-Architecture" (xuyên kiến trúc)** = điểm khó ở đây: teacher và student thuộc **hai họ kiến trúc khác hẳn nhau**:
- Teacher = **ViT** (Vision Transformer) — cắt ảnh thành patch, dùng attention.
- Student = **CNN** (ResNet) — quét ảnh bằng conv, ra feature map dạng lưới.

> **Vấn đề:** CNN và ViT nói "hai ngôn ngữ khác nhau". Không thể copy thẳng đặc trưng của ViT sang CNN. → CAKD gắn thêm vào CNN vài lớp **"phiên dịch"** để nó đẻ ra thứ **có dạng giống ViT**, rồi ép giống teacher. Ngoài ra dùng thêm một **discriminator (GAN)** làm "giám khảo" để tín hiệu bắt chước sắc nét hơn.

---

<a name="2"></a>
## 2. Ba nhân vật

| Nhân vật | Là gì | Trạng thái khi train | Vai trò |
|---|---|---|---|
| **Student** 🎓 | `resnet50_cakd` (ResNet-50 độ thêm) | `.train()` — HỌC | Con model ta muốn dạy cho giỏi. Đầu ra dùng để suy luận thật về sau. |
| **Teacher** 👨‍🏫 | `vit_b_16` (pretrain ImageNet) | `.eval()` — ĐÓNG BĂNG | Thầy giáo. Chỉ chạy xuôi, phát ra "đáp án mẫu". Không bao giờ được cập nhật. |
| **Discriminator** ⚖️ | `NLayerDiscriminator` | `.train()` — HỌC | Giám khảo phân biệt attention "thật" (teacher) vs "giả" (student). |

```
         ┌───────────┐   bắt chước   ┌───────────┐
         │  STUDENT  │ ────────────► │  TEACHER  │
         │  (ResNet) │   (4 loss)    │   (ViT)   │
         └─────┬─────┘               └───────────┘
               │ attention "giả"
               ▼
         ┌───────────────┐
         │ DISCRIMINATOR │  ⚖️ "thật hay giả?"
         └───────────────┘
```

---

<a name="3"></a>
## 3. Bức tranh toàn cảnh 4 file

```
┌──────────────────────────────────────────────────────────────────────┐
│  new_utils.py          → HỘP ĐỒ NGHỀ                                   │
│     GANLoss, NLayerDiscriminator, MetricLogger, accuracy, đa-GPU...    │
├──────────────────────────────────────────────────────────────────────┤
│  cakd_modified_files/resnet.py          → STUDENT (CNN)               │
│     ResNet_CAKD: forward trả 4 thứ (logits, attn, feat, token)        │
├──────────────────────────────────────────────────────────────────────┤
│  cakd_modified_files/vision_transformer.py → TEACHER (ViT)            │
│     VisionTransformer: forward trả 4 thứ (logits, attn, token, feat)  │
├──────────────────────────────────────────────────────────────────────┤
│  dist_train_cakd.py    → NHẠC TRƯỞNG   ★ FILE ĐANG PHÂN TÍCH ★        │
│     Ráp cả 3, tính 4 loss, chạy vòng đấu GAN, lưu checkpoint          │
└──────────────────────────────────────────────────────────────────────┘
```

`dist_train_cakd.py` **không định nghĩa mạng nào** — nó chỉ **gọi ra và điều phối** 3 mảnh trên.

---

<a name="4"></a>
## 4. Kiến trúc tổng thể (sơ đồ lớn)

```
                              Ảnh đầu vào (batch, 3, 224, 224)
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              ▼                                                        ▼
   ╔══════════════════════╗                              ╔══════════════════════╗
   ║   STUDENT  (ResNet)   ║                              ║   TEACHER  (ViT-B/16) ║
   ║   .train() — HỌC      ║                              ║   .eval() — ĐÓNG BĂNG ║
   ╠══════════════════════╣                              ╠══════════════════════╣
   ║ stem → layer1 → layer2║                              ║ cắt 196 patch         ║
   ║ → layer3 (14x14=196)  ║                              ║ + class_token (→197)  ║
   ║      │                ║                              ║ + vị trí (pos_embed)  ║
   ║      ├─ pca_proj  ────╫──► attn_qk, attn_vv          ║      │                ║
   ║      │  (Attention)   ║                              ║  Encoder 12 lớp       ║
   ║      ├─ gl_proj   ────╫──► proj_feat (196,768)       ║      │ (moi attn 2     ║
   ║      │  (GLProj)      ║                              ║      │  lớp cuối)      ║
   ║ → layer4 → avgpool    ║                              ║      ▼                ║
   ║      ├─ fc       ─────╫──► output (logits)           ║ tea_logits            ║
   ║      └─ cls_proj ─────╫──► proj_token (768)          ║ tea_attn_weights (4)  ║
   ╚══════════════════════╝                              ║ tea_token, tea_feat   ║
              │                                          ╚══════════╤═══════════╝
              │                                                     │
              └──────────────────────┬──────────────────────────────┘
                                     ▼
                        ╔════════════════════════╗
                        ║  SO KHỚP → 4 LOSS      ║
                        ║  cls / pca / gl / gan  ║
                        ╚════════════════════════╝
```

---

<a name="5"></a>
## 5. Luồng dữ liệu 1 batch — forward (trong `train_one_epoch`)

Đây là phần **quan trọng nhất**. Mỗi batch đi qua 6 bước:

### Bước 1 — Chạy xuôi cả hai model
```python
output, attn_weights, proj_feat, proj_token = model(image)      # STUDENT → 4 thứ
tea_logits, tea_attn_weights, tea_token, tea_feat = teacher(image)  # TEACHER → 4 thứ
```

| Biến (student) | Ý nghĩa | Biến (teacher) tương ứng |
|---|---|---|
| `output` | logits phân loại | `tea_logits` |
| `attn_weights` = `[attn_qk, attn_vv]` | 2 attention map | `tea_attn_weights` (4 map) |
| `proj_feat` | feature theo patch (đã chiếu 768) | `tea_feat` |
| `proj_token` | class-token (đã chiếu 768) | `tea_token` |

### Bước 2 — Chuẩn bị đầu vào cho Discriminator
```python
input_d_real = tea_attn_weights[2][:, 1:, 1:].clone()[:, None, :, :].detach()  # "THẬT" (teacher)
input_d_fake = attn_weights[0].clone()[:, None, :, :].detach()                 # "GIẢ"  (student)
```
Giải mã dòng khó này:
- `tea_attn_weights[2]` — lấy 1 attention map của teacher.
- `[:, 1:, 1:]` — **cắt bỏ class token** (hàng/cột 0): teacher có 197 token, bỏ token đặc biệt → còn **196×196**, khớp với 196 ô lưới của student.
- `[:, None, :, :]` — chèn 1 chiều "kênh" → shape `(batch, 1, 196, 196)`, đúng định dạng ảnh 1 kênh mà Conv2d của discriminator cần.
- `.detach()` — **cắt gradient**: đây là "mẫu" cố định, không lan ngược vào teacher/student ở bước này.

### Bước 3 — Discriminator chấm điểm
```python
pred_real = discriminator(input_d_real)          # điểm cho attention teacher
pred_fake = discriminator(input_d_fake.detach()) # điểm cho attention student
```

### Bước 4, 5, 6 — Tính 4 loss (xem mục 6) rồi cập nhật trọng số (xem mục 7).

**Sơ đồ forward gọn:**
```
image ──┬──► STUDENT ──► output, attn_qk/vv, proj_feat, proj_token
        │                     │(giả)
        └──► TEACHER ──► tea_logits, tea_attn(4), tea_token, tea_feat
                              │(thật)
              attn_qk & tea_attn ──► DISCRIMINATOR ──► pred_fake / pred_real
```

---

<a name="6"></a>
## 6. 4 hàm loss — giải thích từng cái

### 🔵 cls_loss — Phân loại đúng nhãn thật
```python
cls_loss = criterion(output, target)   # CrossEntropyLoss
```
Nhiệm vụ **gốc** của student: nhìn ảnh đoán đúng lớp. Đây là loss **luôn có** (không nhân hệ số khởi động chậm).

### 🟢 pca_loss — Khớp ATTENTION student ↔ teacher
```python
pca_loss = 0.2 * mse(attn_weights[0], tea_attn_weights[2][:, 1:, 1:].detach()) \
         + 0.05 * mse(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())
```
Ép "bản đồ chú ý" của student giống teacher (đo bằng MSE — sai số bình phương). Dùng 2 cặp attention với trọng số 0.2 và 0.05. `.detach()` phía teacher vì teacher không học.

### 🟡 gl_loss — Khớp LOGITS + TOKEN + FEATURE
```python
gl_loss = mse(output, tea_logits.detach()) \
        + mse(proj_token, tea_token) \
        + 0.05 * mse(proj_feat, tea_feat.detach())
```
- `mse(output, tea_logits)` — KD "mềm": học cả cách teacher phân bố xác suất, không chỉ nhãn cứng.
- `mse(proj_token, tea_token)` — khớp "class token".
- `mse(proj_feat, tea_feat)` — khớp feature theo từng patch.

### 🔴 gan_loss — Đối kháng (dạy DISCRIMINATOR)
```python
gan_loss = 0.5 * (gan_criterion(pred_real.detach(), True) + gan_criterion(pred_fake, False))
```
Dạy discriminator: attention teacher → **True** (thật), attention student → **False** (giả). Đây là loss để cập nhật **discriminator**, KHÔNG phải student.

### Tổng loss cho STUDENT
```python
loss = cls_loss + λ(epoch) * ( pca_loss + gl_loss
                             + 0.05 * gan_criterion(pred_real.detach(), True)
                             +        gan_criterion(pred_fake, True) )   # ← chú ý: True!
```
Phần `gan_criterion(pred_fake, True)` = student **muốn discriminator tưởng attention giả của nó là THẬT** → tức là **đánh lừa** giám khảo. Đây là sức ép khiến student vẽ attention ngày càng giống teacher.

**Bảng tóm tắt:**

| Loss | Đo cái gì | Hàm | Cập nhật ai | Hệ số λ(epoch)? |
|---|---|---|---|---|
| `cls_loss` | phân loại vs nhãn | CrossEntropy | student | ❌ luôn đầy đủ |
| `pca_loss` | attention khớp teacher | MSE | student | ✅ có |
| `gl_loss`  | logits/token/feat khớp teacher | MSE | student | ✅ có |
| `gan_loss` | thật/giả (dạy giám khảo) | GANLoss | **discriminator** | ❌ |
| (phần GAN trong `loss`) | student đánh lừa giám khảo | GANLoss | student | ✅ có |

---

<a name="7"></a>
## 7. Vòng đấu GAN: 2 optimizer chạy xen kẽ

Mỗi batch có **2 bước cập nhật** riêng biệt:

```
        ┌─────────────────────────────────────────────────────────┐
        │  BƯỚC 1 — DẠY DISCRIMINATOR                              │
        │  d_optimizer.zero_grad()                                │
        │  gan_loss.backward(retain_graph=True)  ← giữ đồ thị lại │
        │  d_optimizer.step()                                     │
        │  → Giám khảo giỏi hơn ở việc bắt "attention giả"        │
        ├─────────────────────────────────────────────────────────┤
        │  BƯỚC 2 — DẠY STUDENT                                    │
        │  optimizer.zero_grad()                                  │
        │  loss.backward()                                        │
        │  optimizer.step()                                       │
        │  → Student giỏi hơn ở: phân loại + bắt chước + đánh lừa │
        └─────────────────────────────────────────────────────────┘
```

**Tại sao `retain_graph=True`?** Vì `gan_loss` và `loss` dùng chung một phần đồ thị tính toán (cùng `pred_fake`). Backward lần 1 (gan_loss) mặc định sẽ **xóa đồ thị**; cờ này giữ lại để backward lần 2 (loss) còn dùng được.

**Cuộc đấu:**
```
   Discriminator mạnh lên ──► bắt lỗi attention student gắt hơn
              ▲                              │
              │                              ▼
   Student buộc vẽ attention  ◄──── để đánh lừa được giám khảo
   giống teacher hơn
```
Đây là ý tưởng GAN kinh điển: hai bên "đối kháng" khiến cả hai cùng giỏi lên, kết quả là attention student hội tụ về giống teacher một cách **sắc nét** (hơn là chỉ ép MSE đơn thuần).

> **Chi tiết cân bằng:** `d_optimizer` có learning rate = `0.01 * lr` (nhỏ hơn 100 lần). Cố ý cho discriminator học **chậm** để nó không "quá mạnh" áp đảo student — một mẹo giữ GAN ổn định.

---

<a name="8"></a>
## 8. Cơ chế "khởi động chậm" λ(epoch)

```python
λ = min(max(epoch - 25, 0) / 50.0, 0.2)
```

Vẽ theo epoch:

```
 λ
0.2 ┤                                  ┌────────────────  (chặn trần 0.2)
    │                              ┌───┘
    │                          ┌───┘   ← tăng tuyến tính
    │                      ┌───┘
0.0 ┼──────────────────┬───┘
    └──────────────────┴───────────────────────────────► epoch
    0                 25          35   75

 Giai đoạn:
   epoch 0–25 : λ = 0     → CHỈ học phân loại (cls_loss). Bỏ qua distill.
   epoch 25–75: λ tăng dần 0 → 0.2 (mỗi epoch +1/50)
   epoch >75  : λ = 0.2   → distill ở mức tối đa
```

**Tại sao?** 25 epoch đầu student còn "non", nếu ép nó bắt chước teacher ngay sẽ làm nhiễu việc học cơ bản. Cho nó học phân loại vững trước, **rồi mới** tăng dần sức ép distill. Trần 0.2 để `cls_loss` (nhiệm vụ chính) luôn chiếm ưu thế.

---

<a name="9"></a>
## 9. Luồng train toàn cục (vòng epoch) — hàm `main`

```
main(args)
  │
  ├─ init_distributed_mode      # thiết lập đa GPU
  ├─ load_data                  # nạp ImageNet + augment + sampler
  │
  ├─ TẠO 3 MODEL:
  │     model = resnet50_cakd(...)            # student
  │     teacher = vit_b_16(pretrain)          # teacher
  │     discriminator = NLayerDiscriminator   # giám khảo
  │
  ├─ TẠO 3 LOSS:
  │     criterion = CrossEntropyLoss          # cls
  │     mse_criterion = MSELoss               # pca + gl
  │     gan_criterion = GANLoss               # gan
  │
  ├─ TẠO 2 OPTIMIZER + 2 SCHEDULER:
  │     optimizer   (student)     + lr_scheduler
  │     d_optimizer (discriminator, lr*0.01) + d_lr_scheduler
  │
  ├─ (tùy chọn) EMA, resume checkpoint, DDP wrap
  │
  └─ VÒNG LẶP EPOCH:  for epoch in range(...):
         │
         ├─ train_one_epoch(...)   ← toàn bộ mục 5–8 diễn ra ở đây
         ├─ lr_scheduler.step()    ← giảm lr student
         ├─ d_lr_scheduler.step()  ← giảm lr discriminator
         ├─ evaluate(...)          ← đo accuracy trên tập test
         └─ save checkpoint        ← lưu model (mỗi epoch + bản mốc mỗi 10 epoch)
```

**Sơ đồ thời gian một epoch:**
```
[epoch bắt đầu]
   └─► for mỗi batch:
          forward student+teacher → 4 loss
          → BƯỚC 1: cập nhật discriminator
          → BƯỚC 2: cập nhật student
          → (đôi khi) cập nhật EMA
          → log số liệu
   └─► giảm learning rate
   └─► đánh giá accuracy
   └─► lưu checkpoint
[epoch kết thúc]
```

---

<a name="10"></a>
## 10. Bảng shape tensor qua từng bước (batch = N)

Giả sử ảnh 224×224, teacher ViT-B/16 (patch 16 → 14×14=196 patch, hidden=768).

| Tensor | Shape | Ghi chú |
|---|---|---|
| `image` | `(N, 3, 224, 224)` | ảnh đầu vào |
| student `output` | `(N, num_classes)` | logits |
| student `attn_weights[0]` (attn_qk) | `(N, 196, 196)` | attention map |
| student `proj_feat` | `(N, 196, 768)` | feature theo patch, đã chiếu 768 |
| student `proj_token` | `(N, 768)` | class-token đã chiếu |
| teacher `tea_logits` | `(N, num_classes)` | logits teacher |
| teacher `tea_attn_weights[2]` | `(N, 197, 197)` | có class token |
| `tea_attn_weights[2][:, 1:, 1:]` | `(N, 196, 196)` | bỏ class token → khớp student |
| teacher `tea_token` | `(N, 768)` | class token teacher |
| teacher `tea_feat` | `(N, 196, 768)` | feature patch teacher |
| `input_d_real / input_d_fake` | `(N, 1, 196, 196)` | thêm chiều kênh cho Conv2d |
| `pred_real / pred_fake` | `(N, 1, ...)` | điểm discriminator |

> **Con số then chốt:** `768` (chiều teacher) và `196` (số patch). Student phải chiếu về đúng 2 con số này thì mới so khớp được. Đó chính là lý do tồn tại của `pca_proj`, `gl_proj`, `cls_proj`.

---

<a name="11"></a>
## 11. Bảng đối chiếu Teacher ↔ Student (khớp 1-1)

```
   STUDENT (ResNet_CAKD)          KHỚP QUA          TEACHER (ViT)
   ─────────────────────      ───────────────      ──────────────
   output (logits)      ◄──── gl_loss (MSE) ────►  tea_logits
                        ◄──── cls_loss vs nhãn thật (chỉ student)
   attn_qk, attn_vv     ◄──── pca_loss (MSE) ───►  tea_attn_weights[2],[3]
                        ◄──── gan_loss (đối kháng) qua discriminator
   proj_token (cls_proj)◄──── gl_loss (MSE) ────►  tea_token
   proj_feat (gl_proj)  ◄──── gl_loss (MSE) ────►  tea_feat
```

---

<a name="12"></a>
## 12. Những chỗ dễ nhầm (FAQ)

**❓ Tại sao teacher `.eval()` và dùng `.detach()` khắp nơi?**
Teacher chỉ là "sách giải" — nó đã giỏi sẵn, không được học thêm. `.eval()` tắt dropout/cố định BatchNorm; `.detach()` cắt gradient để không có tín hiệu nào lan ngược làm thay đổi teacher.

**❓ Vì sao attention teacher là `[:, 1:, 1:]` mà student thì không?**
Teacher ViT có thêm 1 "class token" ở đầu → 197 token. Student CNN không có token đó → chỉ 196 ô lưới (14×14). Phải cắt token đầu của teacher để 2 bên **cùng 196×196** mới so khớp được.

**❓ `gan_criterion(pred_fake, False)` và `gan_criterion(pred_fake, True)` — sao cùng `pred_fake` mà lúc False lúc True?**
- Trong `gan_loss` (dạy discriminator): student là **giả** → nhãn `False`. Giám khảo học "đây là giả".
- Trong `loss` (dạy student): student **muốn bị nhìn là thật** → nhãn `True`. Student học "làm sao để giám khảo tưởng mình thật".
Đây chính là hai phía đối lập của cuộc đấu GAN.

**❓ `find_unused_parameters=True` để làm gì?**
Student có nhánh distill (`pca_proj`, `gl_proj`) mà không phải batch nào cũng "chạm" tới hết mọi tham số. Khi train đa GPU (DDP), nếu có tham số không nhận gradient, DDP sẽ báo lỗi trừ khi bật cờ này.

**❓ Sau khi train xong dùng gì để suy luận?**
Chỉ dùng **student** (và chỉ lấy `output` = logits). Teacher và discriminator chỉ là "giàn giáo" lúc train, bỏ đi sau khi xong. Đó là mục tiêu của KD: có được một model **nhỏ mà giỏi**.

**❓ EMA là gì, có bắt buộc không?**
`ExponentialMovingAverage` giữ một bản sao trọng số student được "làm mượt" theo thời gian, thường cho accuracy cao & ổn định hơn khi đánh giá. Không bắt buộc (bật bằng cờ `--model-ema`).

---

## Tóm tắt một câu

> **CAKD** dạy một CNN nhỏ (student) bắt chước một ViT lớn (teacher) bằng cách gắn thêm lớp "phiên dịch" vào CNN để nó đẻ ra attention/feature giống ViT, rồi ép giống teacher qua **4 loss** (phân loại + khớp attention + khớp feature + đối kháng GAN), trong đó một **discriminator** đóng vai giám khảo để tín hiệu bắt chước sắc nét hơn, và sức ép distill được **tăng dần** sau 25 epoch đầu.
