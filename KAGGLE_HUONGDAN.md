# So sánh 3 mốc trên Kaggle (data 3 lớp): ResNet18 baseline → Teacher ViT → CAKD (KD) — copy từng ô

> **Bản port torch 2.x — chạy native, KHÔNG cần ghim torch 1.12, KHÔNG cần chép đè file torch.**
>
> Dataset **3 lớp** glass / paper / plastic (`15k-image-trash`, cấu trúc `class/split/images`).
> Luồng chạy tuần tự **3 mốc để thấy KD có ăn thua không**, mỗi mốc xong ra ngay **biểu đồ + metric + confusion matrix**:
> 1. **ResNet18 baseline** — train thường, KHÔNG teacher — **10% data**.
> 2. **Teacher ViT-B/16** — thầy — **100% data**.
> 3. **CAKD (KD)** — ResNet18 học từ ViT (distillation) — **10% data** (bằng baseline để so công bằng).
>
> Cuối có **1 ô so sánh** cả 3 (bảng + biểu đồ cột). Baseline và CAKD **cùng backbone ResNet18, cùng 10% data**
> → chênh lệch chính là do KD.
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
> Nhánh `mobilenetv3-small` chứa đủ code (resnet18/50/mobilenet + eval + biểu đồ). Ta chạy ResNet18 qua cờ `--student-arch resnet18`.

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
!python -c "import torch; from models.resnet_cakd import resnet18_cakd; from models.vit_cakd import build_teacher; x=torch.randn(2,3,224,224).cuda(); s=resnet18_cakd(num_classes=3).cuda().eval(); o=s(x); t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); print('student', tuple(o[0].shape), '| teacher', tuple(to[0].shape), '| params(M)', round(sum(p.numel() for p in s.parameters())/1e6,1), '| OK')"
```
→ Phải in `student (2, 3) | teacher (2, 3) | params(M) 15.0 | OK`.

---

# 1️⃣ RESNET18 BASELINE (không KD — 10% data)

## Ô 6 — Train baseline ResNet18
```bash
!python /kaggle/working/repo/kaggle_train_resnet18_baseline.py \
  --data-path /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --epochs 30 --batch-size 32 --lr 0.01 \
  --data-fraction 0.1 \
  --output-dir /kaggle/working/baseline_resnet18
```
→ Log in `train: chi dung 10% (...)`. Sinh `baseline_resnet18/resnet18_baseline_best.pth` + `history_resnet18_baseline.json`.

## Ô 7 — Biểu đồ train baseline
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/baseline_resnet18/history_resnet18_baseline.json \
  --out /kaggle/working/plot_baseline.png --title "ResNet18 baseline (10% data)"
from IPython.display import Image, display
display(Image('/kaggle/working/plot_baseline.png'))
```

## Ô 8 — Metric + ma trận nhầm lẫn (baseline)
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py --model baseline --student-arch resnet18 \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/baseline_resnet18/resnet18_baseline_best.pth \
  --out-dir /kaggle/working/results
from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_baseline_resnet18.png'))
```
→ Lưu `results/metrics_baseline_resnet18.json` + confusion matrix.

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
→ 100% data (không cờ `--data-fraction`). Sinh `/kaggle/working/teacher_3cls.pth` + `history_teacher.json`. **Cần teacher tốt trước khi sang mốc 3.**

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

# 3️⃣ CAKD — KD: ViT → ResNet18 (10% data)

## Ô 12 — Distill (train student CAKD, ResNet18, 10% data)
> Bám công thức gốc (auto-augment ta_wide, random-erase 0.1, mixup 0.2, label-smoothing 0.1, wd 2e-5, norm-wd 0,
> ra-sampler reps 4, model-ema, λ khởi động ep25 ramp 50). **1 GPU:** `--nproc_per_node=1`, `--lr 0.0125`.
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch resnet18 --student-pretrained --workers 4 \
  --data-fraction 0.1 \
  --batch-size 32 --lr 0.0125 \
  --lr-warmup-epochs 5 --lr-warmup-method linear \
  --auto-augment ta_wide --epochs 120 --random-erase 0.1 --mixup-alpha 0.2 \
  --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 \
  --distill-start 25 --distill-ramp 50 \
  --train-crop-size 224 --val-resize-size 224 \
  --model-ema --ra-sampler --ra-reps 4 --amp \
  --output-dir /kaggle/working/results
```
→ Log in `pca_loss / gl_loss / cls_loss / gan_loss`. Sinh `results/checkpoint.pth` + `results/history_cakd.json`.

> ⚠️ Muốn nhanh hơn nữa: giảm `--epochs 40 --distill-start 8 --distill-ramp 16 --lr 0.01`.

## Ô 13 — Biểu đồ train CAKD
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/results/history_cakd.json \
  --out /kaggle/working/plot_cakd.png --title "CAKD student ResNet18 (10% data)"
from IPython.display import Image, display
display(Image('/kaggle/working/plot_cakd.png'))
```
- 4 panel: Loss tổng, các loss thành phần `cls/pca/gl/gan` (log), Accuracy (best %), Learning rate (log).

## Ô 14 — Metric + ma trận nhầm lẫn (student CAKD)
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py --model student --student-arch resnet18 --weights ema \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/results/checkpoint.pth \
  --out-dir /kaggle/working/results
from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_student_resnet18.png'))
```
→ Lưu `results/metrics_student_resnet18.json`. `--weights ema` khớp best lúc train (có `--model-ema`).

---

# 4️⃣ SO SÁNH CẢ 3 MỐC

## Ô 15 — Bảng + biểu đồ cột so sánh
```python
!python /kaggle/working/repo/tools/compare_metrics.py \
  --metrics /kaggle/working/results/metrics_baseline_resnet18.json \
            /kaggle/working/results/metrics_teacher.json \
            /kaggle/working/results/metrics_student_resnet18.json \
  --out /kaggle/working/compare.png
from IPython.display import Image, display
display(Image('/kaggle/working/compare.png'))
```
- In **bảng**: accuracy, macro-P/R/F1 + F1 từng lớp cho cả 3 model.
- Biểu đồ: (trái) Accuracy & macro-F1 mỗi model; (phải) F1 từng lớp nhóm theo model.
- **Đọc kết quả:** nếu **ResNet18 (CAKD) > ResNet18 (baseline)** (cùng 10% data) → KD ăn thua.
  So với **ViT (teacher, 100% data)** để thấy student nén được bao nhiêu % kiến thức của thầy dù nhẹ hơn ~5×.

---

## Ghi chú kỹ thuật
- **3 mốc dùng chung tập test** `data_test/val` + cùng preprocessing trong `eval_metrics.py` → so sánh công bằng.
- **Đổi % data:** cờ `--data-fraction` (0.1 = 10%, 0.25 = 1/4, 1.0 = full), lấy stratified chia đều theo lớp, val giữ nguyên.
  Có ở cả `kaggle_train_resnet18_baseline.py`, `dist_train_teacher.py`, `dist_train_cakd.py`.
- **Đổi backbone student:** `--student-arch resnet18|resnet50|mobilenetv3_small` (nhớ đổi khớp ở ô train, eval, biểu đồ).
- **λ (lịch distill):** `min(max(epoch-25,0)/50, 0.2)` qua `--distill-start 25 --distill-ramp 50`.

## Sự cố thường gặp
- **`load_state_dict` lỗi khi eval** → sai `--model`/`--student-arch`: baseline↔`--model baseline --student-arch resnet18`,
  student CAKD↔`--model student --student-arch resnet18`, teacher↔`--model teacher`.
- **Lỗi shape ở `gl_loss`** → teacher chưa 3 lớp: kiểm tra `--teacher-weights` trỏ đúng `teacher_3cls.pth`.
- **`CUDA out of memory`** → giảm `--batch-size` 16/8, giữ `--amp`.
- **`... modified by an inplace operation`** → chạy code cũ: `!cd /kaggle/working/repo && git pull`.
- **Train "không thấy log gì"** → KHÔNG treo; lần đầu mất vài phút quét ảnh + tải backbone pretrained + dựng sampler.
  Đã có `PYTHONUNBUFFERED=1` để log hiện ngay.

## Chạy thử nhanh trước khi chạy full
```bash
# baseline thử (2 epoch, 10%)
!python /kaggle/working/repo/kaggle_train_resnet18_baseline.py --epochs 2 --data-fraction 0.1 \
  --output-dir /kaggle/working/baseline_resnet18
# teacher thử (2 epoch, full)
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if --batch-size 16 --epochs 2 --amp --output-dir /kaggle/working
# CAKD thử (2 epoch, 10%)
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch resnet18 --student-pretrained --workers 4 --batch-size 16 --lr 0.0125 --epochs 2 \
  --data-fraction 0.1 --distill-start 0 --distill-ramp 2 --amp --output-dir /kaggle/working/results
```
