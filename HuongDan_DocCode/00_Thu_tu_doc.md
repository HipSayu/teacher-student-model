# 🗺️ Lộ trình đọc code CAKD — đọc theo đúng thứ tự này

> Mục tiêu: đọc từ **viên gạch nhỏ → mô hình → vòng train**, để khi tới file train mày đã hiểu hết
> các mảnh ghép, không bị "nhảy cóc".
>
> Ký hiệu trong bộ tài liệu:
> - 🟢 **ĐỌC KỸ** — code CAKD **tự viết**, phải hiểu từng dòng.
> - ⚪ **LƯỚT** — lấy nguyên từ torchvision, đọc để biết nó tồn tại, không cần soi từng dòng.

---

## Thứ tự đọc (5 chặng)

| # | File | Đọc phần nào | Tài liệu kèm |
|---|------|--------------|--------------|
| **1** | `CAKD/new_utils.py` | 🟢 `GANLoss` (20–79), `NLayerDiscriminator` (81–126) · ⚪ phần còn lại | [01_new_utils.md](01_new_utils.md) |
| **2** | `CAKD/cakd_modified_files/resnet.py` | 🟢 `Attention` (90–121), `GLProj` (123–152), `ResNet_CAKD` (420–553), `resnet50_cakd` (1022+) · ⚪ `BasicBlock`/`Bottleneck`/`ResNet`/các `*_Weights` | [02_resnet_cakd.md](02_resnet_cakd.md) |
| **3** | `CAKD/cakd_modified_files/vision_transformer.py` | 🟢 `EncoderBlock.forward` (110–121), `Encoder.forward` (156–169), `VisionTransformer.forward` (301–318) · ⚪ phần còn lại | [03_vit_teacher.md](03_vit_teacher.md) |
| **4** | `CAKD/dist_train_cakd.py` | 🟢 **TẤT CẢ** — đặc biệt `train_one_epoch` (18–85) · ⚪ `get_args_parser` (440–567) | [04_dist_train_cakd.md](04_dist_train_cakd.md) |
| **5** | `CAKD/transforms.py` · `CAKD/dist_train_logits.py` | ⚪ đọc tham khảo (Mixup/CutMix; biến thể chỉ-logits) | mục cuối file này |

---

## Vì sao đọc theo thứ tự này?

```
      CHẶNG 1                 CHẶNG 2 + 3               CHẶNG 4
  ┌─────────────┐        ┌──────────────────┐     ┌──────────────┐
  │  new_utils  │        │  resnet_cakd     │     │ dist_train   │
  │  ─────────  │───────►│  (HỌC SINH)      │────►│ ─ ghép tất cả│
  │ GANLoss     │ dùng   │  vit (GIÁO VIÊN) │ tạo │ ─ train loop │
  │ Discriminator│  bởi  │                  │ ra  │ ─ 4 loss     │
  └─────────────┘        └──────────────────┘     └──────────────┘
     viên gạch              các mô hình              nhà hoàn chỉnh
```

1. **Chặng 1 — viên gạch nhỏ:** `GANLoss` và `Discriminator` là 2 lớp độc lập, không phụ thuộc gì.
   Hiểu trước thì tới vòng train sẽ không bỡ ngỡ khi thấy `gan_criterion(...)` và `discriminator(...)`.
2. **Chặng 2 — học sinh:** hiểu `model(image)` trả về **4 thứ** (`output, [attn_qk,attn_vv], proj_feat, proj_token`)
   và chúng sinh ra từ đâu (nhánh rẽ tại `layer3`).
3. **Chặng 3 — giáo viên:** hiểu `teacher(image)` trả về **4 thứ** tương ứng, và vì sao chỉ lấy attention của 2 lớp cuối.
4. **Chặng 4 — ghép tất cả:** giờ đọc `train_one_epoch` thì mọi biến đều có nghĩa → thấy được toàn cảnh:
   4 loss lắp thế nào, 2 lần `backward` ra sao.

---

## Bản đồ "ai trả về cái gì" (dán lên tường)

```
model(image)   → output           , [attn_qk, attn_vv] , proj_feat        , proj_token
  (HỌC SINH)      (32, n_cls)        mỗi cái (32,196,196)  (32,196,768)       (32,768)
                  │                  │                     │                  │
                  ▼ cls_loss/gl      ▼ pca_loss + GAN      ▼ gl_loss          ▼ gl_loss
teacher(image) → tea_logits        , tea_attn[0..3]      , tea_token        , tea_feat
  (GIÁO VIÊN)     (32, n_cls)        4 map (qk₂,vv₂,qk₁,vv₁) (32,768)          (32,196,768)
  (đóng băng ❄)   → khớp gl          → khớp pca (dùng [2],[3]) → khớp token    → khớp feat
```

Giữ bản đồ này bên cạnh khi đọc `train_one_epoch` — 90% sự rối rắm đến từ việc quên biến nào là của ai.

---

## Chặng 5 — file phụ (chỉ đọc khi cần)

- **`CAKD/transforms.py`** — `RandomMixup`, `RandomCutmix`. Đây là phần hiện thực **"MVG" (đa khung nhìn)**
  bằng augmentation chuẩn. Đọc nếu muốn hiểu `collate_fn` trong `main()` trộn ảnh/nhãn thế nào.
- **`CAKD/dist_train_logits.py`** — biến thể **chỉ chưng cất logits** (KD kinh điển, không PCA/GL/GAN).
  Dùng để so sánh "CAKD đầy đủ" vs "KD thường". Đọc SAU khi đã hiểu `dist_train_cakd.py` — nó là bản rút gọn.
- **`CAKD/cakd_modified_files/functional.py`** — ⚪ toàn bộ là torchvision gốc, **không cần đọc** trừ khi debug transform.

---

## Cách dùng bộ tài liệu này

Mở **2 cửa sổ song song**: bên trái là file `.py` thật, bên phải là file `.md` hướng dẫn tương ứng.
Mỗi mục trong file `.md` ghi rõ **số dòng** để mày nhảy tới đúng chỗ trong code.
Đọc hết chặng nào thì tự hỏi lại câu "kiểm tra hiểu" ở cuối mỗi file hướng dẫn trước khi sang chặng sau.
