# So sánh 3 mốc trên Kaggle (rác 3 lớp): MobileNetV3 baseline → Teacher ViT → CAKD (KD) — copy từng ô

> **Bản port torch 2.x — chạy native, KHÔNG cần ghim torch 1.12, KHÔNG cần chép đè file torch.**
>
> Dataset **3 lớp** glass / paper / plastic (`15k-image-trash`, cấu trúc `class/split/images`).
> Luồng chạy tuần tự **3 mốc**, mỗi mốc xong ra ngay **biểu đồ + metric + confusion matrix**:
> 1. **MobileNetV3-Small baseline** — train thường, KHÔNG teacher — **10% data**.
> 2. **Teacher ViT-B/16** — thầy — **100% data**.
> 3. **CAKD (KD)** — MobileNetV3-Small học từ ViT (distillation) — **100% data**.
>
> Cuối có **1 ô so sánh** cả 3 (bảng + biểu đồ cột).
>
> ⚠️ **Lưu ý khi đọc kết quả:** baseline chạy 10% data còn KD chạy 100% data → nếu KD cao hơn thì **phần thắng
> đến từ CẢ hai yếu tố** (nhiều data hơn + có KD), không tách riêng được công của KD. Muốn chứng minh riêng
> tác dụng của KD thì cho baseline và KD **cùng %** data (đổi `--data-fraction` cho khớp).
>
> Chuẩn bị: New Notebook → **Settings → Accelerator → GPU T4/P100** → Add Input: thêm dataset `15k-image-trash`.

---

# ⚙️ SETUP (chạy 1 lần)

## Ô 1 — Xác nhận đường dẫn dataset
```bash
!ls /kaggle/input
!ls /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image
```
→ Lệnh thứ 2 phải in `glass  paper  plastic`. Nếu path khác, sửa lại `--data-path`/`--src` ở các ô dưới.

## Ô 2 — Clone code + cài einops
```bash
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git fetch origin && git checkout mobilenetv3-small && git pull) \
  || git clone -b mobilenetv3-small https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops
```

## Ô 3 — Reorg ảnh → ImageFolder (cho teacher + CAKD)
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_if \
  --classes glass paper plastic --splits train val test
```

## Ô 4 — Chuẩn bị tập TEST để đánh giá (dùng chung cho cả 3 mốc)
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
```
> Cả 3 mốc chấm trên **cùng tập test này** (`data_test/val`) → so sánh công bằng.

## Ô 5 — Verify model dựng đúng
```bash
%cd /kaggle/working/repo/CAKD
!python -c "import torch; from models.mobilenet_cakd import mobilenetv3_small_cakd; from models.vit_cakd import build_teacher; x=torch.randn(2,3,224,224).cuda(); s=mobilenetv3_small_cakd(num_classes=3).cuda().eval(); o=s(x); t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); print('student', tuple(o[0].shape), '| teacher', tuple(to[0].shape), '| params(M)', round(sum(p.numel() for p in s.parameters())/1e6,2), '| OK')"
```
→ Phải in `student (2, 3) | teacher (2, 3) | params(M) 2.58 | OK`.

---

# 1️⃣ MOBILENETV3-SMALL BASELINE (không KD — 10% data)

## Ô 6 — Train baseline MobileNetV3
```bash
!python /kaggle/working/repo/kaggle_train_mobilenetv3_baseline.py \
  --data-path /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --epochs 30 --batch-size 32 --lr 0.01 \
  --data-fraction 0.1 \
  --output-dir /kaggle/working/baseline_mobilenetv3
```
→ Log in `Layout: split_images | 3 lop: [...]` và `train: chi dung 10% (...)`.
Sinh `baseline_mobilenetv3/mobilenetv3_baseline_best.pth` + `history_mobilenetv3_baseline.json`.

## Ô 7 — Biểu đồ train baseline
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/baseline_mobilenetv3/history_mobilenetv3_baseline.json \
  --out /kaggle/working/plot_baseline.png --title "MobileNetV3 baseline (10% data)"
from IPython.display import Image, display
display(Image('/kaggle/working/plot_baseline.png'))
```

## Ô 8 — Metric + ma trận nhầm lẫn (baseline)
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py --model baseline --student-arch mobilenetv3_small \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/baseline_mobilenetv3/mobilenetv3_baseline_best.pth \
  --out-dir /kaggle/working/results
from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_baseline_mobilenetv3_small.png'))
```
→ Lưu `results/metrics_baseline_mobilenetv3_small.json` + confusion matrix.

---

# 2️⃣ TEACHER ViT-B/16 (100% data)

## Ô 9 — Fine-tune teacher ViT → 3 lớp
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 20 --lr 2e-4 --wd 0.05 --label-smoothing 0.1 \
  --amp --output-dir /kaggle/working
```
→ **100% data** (không có cờ `--data-fraction` nên mặc định 1.0 = full).
Sinh `/kaggle/working/teacher_3cls.pth` + `history_teacher.json`. **Cần teacher tốt trước khi sang mốc 3.**

## Ô 10 — Biểu đồ train teacher
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/history_teacher.json \
  --out /kaggle/working/plot_teacher.png --title "Teacher ViT-B/16 (100% data)"
from IPython.display import Image, display
display(Image('/kaggle/working/plot_teacher.png'))
```

## Ô 11 — Metric + ma trận nhầm lẫn (teacher)
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py --model teacher \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/teacher_3cls.pth \
  --out-dir /kaggle/working/results
from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_teacher.png'))
```
→ Lưu `results/metrics_teacher.json` + confusion matrix.

---

# 3️⃣ CAKD — KD: ViT → MobileNetV3-Small (100% data)

## Ô 12 — Distill (train student CAKD)
> Bám công thức gốc (auto-augment ta_wide, random-erase 0.1, mixup 0.2, label-smoothing 0.1, wd 2e-5, norm-wd 0,
> ra-sampler reps 4, model-ema, λ khởi động ep25 ramp 50). **1 GPU:** `--nproc_per_node=1`, `--lr 0.0125`.
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch mobilenetv3_small --student-pretrained --workers 4 \
  --batch-size 32 --lr 0.0125 \
  --lr-warmup-epochs 5 --lr-warmup-method linear \
  --auto-augment ta_wide --epochs 120 --random-erase 0.1 --mixup-alpha 0.2 \
  --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 \
  --distill-start 25 --distill-ramp 50 \
  --train-crop-size 224 --val-resize-size 224 \
  --model-ema --ra-sampler --ra-reps 4 --amp \
  --output-dir /kaggle/working/results
```
→ **100% data** (không có `--data-fraction`). Log in `pca_loss / gl_loss / cls_loss / gan_loss`.
Sinh `results/checkpoint.pth` + `results/history_cakd.json`.

> ⚠️ 120 epoch trên full data / 1 GPU khá lâu (nhiều giờ). Muốn nhanh:
> `--epochs 40 --distill-start 8 --distill-ramp 16 --lr 0.01`, hoặc thêm `--data-fraction 0.25`.
> Backbone nhẹ nên nếu loss khó xuống có thể hạ `--lr 0.008`.

## Ô 13 — Biểu đồ train CAKD
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/results/history_cakd.json \
  --out /kaggle/working/plot_cakd.png --title "CAKD student MobileNetV3 (100% data)"
from IPython.display import Image, display
display(Image('/kaggle/working/plot_cakd.png'))
```
- 4 panel: Loss tổng, các loss thành phần `cls/pca/gl/gan` (log), Accuracy (best %), Learning rate (log).

## Ô 14 — Metric + ma trận nhầm lẫn (student CAKD)
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py --model student --student-arch mobilenetv3_small --weights ema \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/results/checkpoint.pth \
  --out-dir /kaggle/working/results
from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_student_mobilenetv3_small.png'))
```
→ Lưu `results/metrics_student_mobilenetv3_small.json`. `--weights ema` khớp best lúc train (có `--model-ema`).

---

# 4️⃣ SO SÁNH CẢ 3 MỐC

## Ô 15 — Bảng + biểu đồ cột so sánh
```python
!python /kaggle/working/repo/tools/compare_metrics.py \
  --metrics /kaggle/working/results/metrics_baseline_mobilenetv3_small.json \
            /kaggle/working/results/metrics_teacher.json \
            /kaggle/working/results/metrics_student_mobilenetv3_small.json \
  --out /kaggle/working/compare.png
from IPython.display import Image, display
display(Image('/kaggle/working/compare.png'))
```
- In **bảng**: accuracy, macro-P/R/F1 + F1 từng lớp cho cả 3 model.
- Biểu đồ: (trái) Accuracy & macro-F1 mỗi model; (phải) F1 từng lớp nhóm theo model.
- **Đọc kết quả:** so **MobileNet (CAKD, 100%)** với **ViT (teacher, 100%)** → student nén được bao nhiêu %
  kiến thức của thầy dù nhẹ hơn ~33×. Còn **MobileNet (baseline, 10%)** là mốc tham chiếu data ít.

---

## Ghi chú kỹ thuật
- **3 mốc dùng chung tập test** `data_test/val` + cùng preprocessing trong `eval_metrics.py` → so sánh công bằng.
- **Đổi % data:** cờ `--data-fraction` (0.1 = 10%, 0.25 = 1/4, 1.0 = full; **bỏ cờ = full**), lấy stratified chia
  đều theo lớp, val giữ nguyên. Có ở cả `kaggle_train_mobilenetv3_baseline.py`, `dist_train_teacher.py`, `dist_train_cakd.py`.
- **Đổi backbone student:** `--student-arch mobilenetv3_small|resnet18|resnet50` (nhớ đổi khớp ở ô train, eval, biểu đồ).
- **Student** `mobilenetv3_small_cakd` (~2.6M): tách backbone tại tầng 14×14 (48 kênh/196 token) cắm 2 đầu distill
  `pca_proj`/`gl_proj`; `features[9:]`+classifier lo phân loại.
- **λ (lịch distill):** `min(max(epoch-25,0)/50, 0.2)` qua `--distill-start 25 --distill-ramp 50`.

## Sự cố thường gặp
- **`load_state_dict` lỗi khi eval** → sai `--model`/`--student-arch`: baseline↔`--model baseline --student-arch mobilenetv3_small`,
  student CAKD↔`--model student --student-arch mobilenetv3_small`, teacher↔`--model teacher`.
- **Lỗi shape ở `gl_loss`** → teacher chưa 3 lớp: kiểm tra `--teacher-weights` trỏ đúng `teacher_3cls.pth`.
- **`CUDA out of memory`** → giảm `--batch-size` 16/8, giữ `--amp`.
- **`unrecognized arguments: --data-fraction`** → code cũ: `!cd /kaggle/working/repo && git pull`.
- **Train "không thấy log gì"** → KHÔNG treo; lần đầu mất vài phút quét ảnh + tải backbone pretrained + dựng sampler.

## Chạy thử nhanh trước khi chạy full
```bash
# baseline thử (2 epoch, 10%)
!python /kaggle/working/repo/kaggle_train_mobilenetv3_baseline.py --epochs 2 --data-fraction 0.1 \
  --output-dir /kaggle/working/baseline_mobilenetv3
# teacher thử (2 epoch)
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if --batch-size 16 --epochs 2 --amp --output-dir /kaggle/working
# CAKD thử (2 epoch, 25% cho nhanh)
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch mobilenetv3_small --student-pretrained --workers 4 --batch-size 16 --lr 0.0125 --epochs 2 \
  --data-fraction 0.25 --distill-start 0 --distill-ramp 2 --amp --output-dir /kaggle/working/results
```
