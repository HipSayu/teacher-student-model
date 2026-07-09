# Thiết kế: Huấn luyện CAKD phân loại 3 lớp (glass / paper / plastic) trên Kaggle

- **Ngày:** 2026-07-09
- **Người dùng:** Hieppn
- **Trạng thái:** Chờ duyệt

---

## 1. Mục tiêu

Huấn luyện student **ResNet-50** phân loại **3 lớp: glass, paper, plastic** bằng phương pháp
**CAKD** (Cross-Architecture Knowledge Distillation) — chưng cất tri thức từ teacher **ViT-B/16**
sang student ResNet-50, trên môi trường **Kaggle Notebook (1 GPU)**.

Kết quả cuối cùng dùng để suy luận: **chỉ student ResNet-50** (lấy nhánh logits).

### Quyết định đã chốt với người dùng
| Vấn đề | Quyết định |
|---|---|
| Số lớp | **3**: glass, paper, plastic (bỏ metal) |
| Định dạng dữ liệu nguồn | YOLO: mỗi class 1 thư mục → `train/val/test` → `images/` + `labels/` |
| Hướng xử lý teacher | **Hướng A** — fine-tune ViT teacher xuống 3 lớp trước, rồi mới CAKD |
| Khởi tạo student | **Pretrained ImageNet backbone** (ResNet-50) |
| Môi trường | **Kaggle** (1 GPU) |

---

## 2. Bối cảnh & các ràng buộc kỹ thuật đã kiểm chứng trong code

1. **`num_classes` lấy tự động** từ `ImageFolder`: `dist_train_cakd.py:378` `num_classes = len(dataset.classes)`.
   → Chỉ cần dữ liệu đúng cấu trúc thư mục theo lớp là ra 3 lớp.

2. **Teacher ViT pretrain bị ép 1000 lớp:** `vision_transformer.py:400` gọi
   `_ovewrite_named_param(kwargs, "num_classes", 1000)` và `load_state_dict` head 1000 lớp.
   → Nếu distill logits trực tiếp, `gl_loss` (`dist_train_cakd.py:130` `mse(output[N,3], tea_logits[N,1000])`)
   **vỡ shape**. Đây là lý do phải fine-tune teacher xuống 3 lớp (Hướng A).

3. **Student pretrained nạp `strict=False`:** `resnet.py:688`. Khi truyền `weights`, hàm ép
   `num_classes=1000` → sau khi tạo model phải **thay `model.fc = nn.Linear(2048, 3)`**.
   Các lớp distill `pca_proj/gl_proj/cls_proj` giữ khởi tạo ngẫu nhiên (đúng thiết kế).

4. **`GLProj.forward` hardcode `.to('cuda')`** (`resnet.py:207`): model **bắt buộc chạy trên GPU**.
   Khớp môi trường Kaggle GPU.

5. **Các file "độ" dựa trên torch 1.12:** README yêu cầu chép đè `resnet.py`,
   `vision_transformer.py` (vào `torchvision/models/`) và `functional.py` (vào `torch/nn/`).
   `functional.py` phụ thuộc phiên bản torch → **phải ghim `torch==1.12.0` + `torchvision==0.13.0`**
   để 3 file này khớp API nội bộ. Chép đè `functional.py` 1.12 vào torch 2.x sẽ hỏng.

6. **Script gốc chạy 8 GPU qua `torchrun --nproc_per_node=8` + DDP.** Trên Kaggle 1 GPU dùng
   `torchrun --nproc_per_node=1`; hàm `evaluate` có tham chiếu `torch.distributed.get_rank()`
   (được `and` short-circuit bảo vệ, nhưng chạy qua torchrun 1 process là an toàn nhất).

7. **`λ(epoch) = min(max(epoch-25,0)/50, 0.2)`** (`dist_train_cakd.py:146`): 25 epoch đầu KHÔNG distill.
   Lịch này thiết kế cho ~120 epoch ImageNet; với dataset nhỏ + số epoch ít hơn cần **nén lại lịch**.

---

## 3. Kiến trúc giải pháp (tổng thể)

Quy trình gồm **4 bước tuần tự**, mỗi bước là một đơn vị độc lập, giao tiếp qua **file trên đĩa**
(dataset đã chuyển đổi, checkpoint teacher). Điều này giúp chạy lại từng bước mà không phải làm lại từ đầu.

```
[B0] Setup môi trường Kaggle
        │  (torch 1.12 + chép 3 file độ + verify import)
        ▼
[B1] Chuyển dữ liệu YOLO ──► ImageFolder
        │  tools/yolo_to_imagefolder.py
        │  data_if/{train,val,test}/{glass,paper,plastic}/*.jpg
        ▼
[B2] GĐ1 — Fine-tune teacher ViT-B/16 xuống 3 lớp
        │  dist_train_teacher.py  ──►  teacher_3cls.pth
        ▼
[B3] GĐ2 — CAKD distill ViT(3 lớp) ──► ResNet-50(3 lớp)
        │  dist_train_cakd.py (đã chỉnh)  ──►  checkpoint.pth (student)
        ▼
[B4] Đánh giá cuối trên tập test + (tùy chọn) suy luận 1 ảnh
```

---

## 4. Chi tiết từng đơn vị

### 4.1. `tools/yolo_to_imagefolder.py` — Chuyển định dạng dữ liệu

**Đầu vào:** thư mục gốc dataset YOLO, ví dụ:
```
<root>/glass/{train,val,test}/images/*.jpg   (+ labels/*.txt bị bỏ qua)
<root>/paper/{train,val,test}/images/*.jpg
<root>/plastic/{train,val,test}/images/*.jpg
```
**Đầu ra:** cấu trúc `ImageFolder`:
```
<out>/train/glass/*.jpg    <out>/val/glass/*.jpg    <out>/test/glass/*.jpg
<out>/train/paper/*.jpg    <out>/val/paper/*.jpg    <out>/test/paper/*.jpg
<out>/train/plastic/*.jpg  <out>/val/plastic/*.jpg  <out>/test/plastic/*.jpg
```

**Logic:**
- Nhãn của ảnh = **tên thư mục lớp gốc** (glass/paper/plastic). Bỏ qua hoàn toàn `labels/*.txt`
  (đó là annotation detection, không cần cho classification).
- Tạo file bằng **copy** (mặc định) hoặc **symlink** (cờ `--symlink`, tiết kiệm dung lượng — Kaggle
  input là read-only nên output ghi sang `/kaggle/working`).
- Tham số: `--src`, `--dst`, `--classes glass paper plastic`, `--splits train val test`, `--symlink`.
- Bỏ qua an toàn nếu thiếu split `test` (chỉ cảnh báo, vẫn tạo train/val).
- In thống kê số ảnh mỗi lớp/mỗi split để người dùng kiểm tra cân bằng dữ liệu.

**Giả định cần người dùng xác nhận khi chạy:** mỗi thư mục lớp gốc chỉ chứa ảnh của đúng lớp đó
(nghĩa là ảnh trong `glass/` đều là thủy tinh). Nếu 1 ảnh chứa nhiều vật thể khác lớp thì cách gán
nhãn theo tên thư mục sẽ sai — nhưng với bộ phân loại vật liệu tái chế điều này thường đúng.

### 4.2. Setup môi trường Kaggle (cell trong notebook `notebooks/kaggle_cakd.ipynb`)

Các bước, đóng thành hàm/cell rõ ràng:
1. `pip install torch==1.12.0+cu113 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113`
   và `pip install einops` (student dùng einops).
2. Định vị đường dẫn cài đặt:
   - `TORCHVISION_MODELS = <torchvision>/models/`
   - `TORCH_NN = <torch>/nn/`
   (lấy qua `torchvision.models.__path__[0]`, `torch.nn.__path__[0]`).
3. **Sao lưu** rồi **chép đè**:
   - `CAKD/cakd_modified_files/resnet.py` → `TORCHVISION_MODELS/resnet.py`
   - `CAKD/cakd_modified_files/vision_transformer.py` → `TORCHVISION_MODELS/vision_transformer.py`
   - `CAKD/cakd_modified_files/functional.py` → `TORCH_NN/functional.py`
4. **Verify** (bắt buộc, nếu fail thì dừng):
   - `import torchvision; torchvision.models.resnet50_cakd` tồn tại.
   - Chạy thử 1 forward: `resnet50_cakd(num_classes=3).cuda()(torch.randn(2,3,224,224).cuda())`
     trả về 4-tuple đúng shape.
   - `vit_b_16(...).cuda()` trả về 4-tuple.

> **Rủi ro & phương án dự phòng:** cài torch 1.12 đè lên torch preinstalled của Kaggle có thể
> chậm hoặc xung đột CUDA driver. Nếu không cài được 1.12: phương án dự phòng là **port 3 file độ
> sang torch hiện hành của Kaggle** (chủ yếu chỉnh phần `multi_head_attention_forward` trong
> `functional.py` và các import nội bộ) — phức tạp hơn, để riêng một mục "nếu cần" trong plan.

### 4.3. `dist_train_teacher.py` — GĐ1: Fine-tune teacher xuống 3 lớp (script mới)

Dựng gọn dựa trên khung `dist_train_logits.py` (đã có sẵn load_data/evaluate) nhưng **bỏ phần distill**:
- Tạo `teacher = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)` (1000 lớp, có pretrain).
- **Thay head:** `teacher.heads.head = nn.Linear(768, 3)` (khởi tạo lại).
- **Đóng băng tùy chọn:** cờ `--freeze-backbone` (mặc định off) — nếu bật, chỉ train head vài epoch
  đầu rồi mở khóa (linear-probe → fine-tune). Đơn giản nhất: fine-tune toàn bộ với lr nhỏ.
- Loss: chỉ `CrossEntropyLoss` trên `out[0]` (vì forward trả 4-tuple, lấy phần tử 0 = logits).
- Optimizer: **AdamW**, lr ~ `1e-4`..`3e-4`, ~10–20 epoch (ViT fine-tune hội tụ nhanh).
- `evaluate` dùng `out[0]`.
- Lưu **`teacher_3cls.pth`** = `state_dict` của teacher (num_classes=3).
- Chạy 1 GPU: `torchrun --nproc_per_node=1 dist_train_teacher.py ...` (hoặc chạy thẳng python nếu
  ta thêm nhánh non-distributed an toàn).

**Tiêu chí thành công B2:** teacher đạt accuracy hợp lý trên tập val (kỳ vọng cao, ViT mạnh) —
đây là "thầy" đủ giỏi để dạy student.

### 4.4. Chỉnh `dist_train_cakd.py` — GĐ2: CAKD distill (sửa file hiện có)

Các thay đổi tối thiểu, có kiểm soát:
1. **Nạp teacher 3 lớp thay vì ImageNet:**
   - Thêm cờ `--teacher-weights <path>` (mặc định `teacher_3cls.pth`).
   - `teacher = vit_b_16()` **không weights** → tạo với `num_classes=3` (truyền `num_classes=num_classes`).
   - `teacher.heads.head = nn.Linear(768, 3)` rồi `teacher.load_state_dict(torch.load(path))`.
   - Bỏ import/dùng `ViT_B_16_Weights.IMAGENET1K_V1` cho teacher.
   Sau thay đổi này, `tea_logits` có shape `[N,3]` → `gl_loss` khớp `output[N,3]` ✔.
2. **Student pretrained backbone:**
   - Thêm cờ `--student-pretrained` (mặc định bật).
   - `model = resnet50_cakd(weights=ResNet50_Weights.IMAGENET1K_V1)` rồi
     `model.fc = nn.Linear(2048, num_classes)` và `model.cls_proj` giữ nguyên.
     (weights nạp strict=False, ép num_classes=1000 nên phải thay fc = 3.)
3. **An toàn 1 GPU:** chạy qua `torchrun --nproc_per_node=1`. Không đổi logic DDP (giữ nguyên
   `find_unused_parameters=True`). Kiểm tra `evaluate`/`reduce_across_processes` chạy ổn với world_size=1.
4. **Nén lịch λ(epoch) cho dataset nhỏ:** thêm 2 cờ `--distill-start` (mặc định 5) và
   `--distill-ramp` (mặc định 20), thay công thức cứng thành
   `λ = min(max(epoch - distill_start, 0) / distill_ramp, 0.2)`. Giữ mặc định tương đương gốc nếu
   người dùng truyền 25 và 50.
5. **KHÔNG đổi** 4 công thức loss, vòng đấu GAN, EMA — giữ nguyên phương pháp CAKD.

### 4.5. Script chạy Kaggle + đánh giá

- `experiments/run_teacher_kaggle.sh` và `experiments/run_cakd_kaggle.sh`: lệnh `torchrun --nproc_per_node=1`
  với siêu tham số phù hợp dataset nhỏ (xem §5).
- Đánh giá cuối trên tập **test**: chạy `dist_train_cakd.py --test-only --resume checkpoint.pth`
  trỏ `val` sang thư mục `test`. Báo cáo Acc@1 (Acc@5 vô nghĩa với 3 lớp, sẽ luôn ~100%, ghi chú rõ).
- (Tùy chọn) ô notebook suy luận 1 ảnh: load student, in nhãn dự đoán + xác suất softmax.

---

## 5. Siêu tham số đề xuất (dataset nhỏ, 1 GPU)

Giá trị khởi điểm, tinh chỉnh theo kích thước dataset thực tế:

| Tham số | GĐ1 (teacher) | GĐ2 (CAKD student) |
|---|---|---|
| epochs | 15 | 60 |
| batch-size | 32 (giảm nếu hết VRAM) | 32 |
| optimizer | AdamW | SGD (giữ như gốc) |
| lr | 2e-4 | 0.01 (giảm từ 0.1 vì batch nhỏ + pretrained) |
| lr-warmup-epochs | 2 | 5 |
| distill-start / ramp | — | 5 / 20 |
| augment | resize/crop/flip nhẹ | `--auto-augment ta_wide --random-erase 0.1 --mixup-alpha 0.2` (như gốc, cân nhắc giảm nếu data quá ít) |
| model-ema | tùy | bật (`--model-ema`) |

> **Lưu ý overfit:** với vài nghìn ảnh, ResNet-50 rất dễ overfit. Ưu tiên pretrained backbone (đã chọn),
> lr nhỏ, augment mạnh vừa phải, theo dõi khoảng cách train/val acc.

---

## 6. Phạm vi & những gì KHÔNG làm (YAGNI)

- **Không** viết lại kiến trúc CAKD, không đổi công thức loss/GAN.
- **Không** hỗ trợ đa GPU mới (giữ code DDP cũ, chỉ chạy 1 process).
- **Không** làm pipeline detection (bỏ qua `labels/`).
- **Không** xây web/app suy luận — chỉ 1 ô notebook demo tùy chọn.
- **Không** tự động tải dataset từ Kaggle API (người dùng đã có sẵn trong `/kaggle/input`).

---

## 7. Rủi ro chính & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Cài `torch==1.12` trên Kaggle thất bại/xung đột | Verify sớm ở B0; phương án B: port 3 file độ sang torch hiện hành |
| Teacher fine-tune kém → distill vô ích | Kiểm tra acc teacher ở B2 trước khi sang B3 |
| Dataset quá nhỏ → student overfit | Pretrained backbone + augment + EMA + theo dõi val |
| Gán nhãn theo tên thư mục sai | In thống kê + người dùng xác nhận mỗi thư mục thuần 1 lớp |
| `evaluate` lỗi với world_size=1 | Luôn chạy qua `torchrun --nproc_per_node=1` để init distributed |
| Hết VRAM (teacher ViT + student + GAN) | Giảm batch-size; bật `--amp` (mixed precision) |

---

## 8. Tiêu chí hoàn thành

1. `tools/yolo_to_imagefolder.py` tạo đúng cấu trúc `ImageFolder` 3 lớp, in thống kê hợp lý.
2. Notebook setup chạy tới bước verify **thành công** (import + forward 4-tuple đúng shape).
3. B2 sinh `teacher_3cls.pth` với accuracy val báo cáo được.
4. B3 chạy trọn vẹn không lỗi shape, sinh `checkpoint.pth` cho student, log 4 loss giảm hợp lý.
5. B4 báo cáo Acc@1 của student trên tập test.
