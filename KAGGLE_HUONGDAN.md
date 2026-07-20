# So sánh 3 mốc trên Kaggle: MobileNetV3 baseline → Teacher ViT → CAKD (KD) — copy từng ô

> **Bản port torch 2.x — chạy native, KHÔNG cần ghim torch 1.12, KHÔNG cần chép đè file torch.**
>
> Luồng này chạy tuần tự **3 mốc để thấy KD có ăn thua không**, mỗi mốc train xong ra ngay **biểu đồ + metric + ma trận nhầm lẫn**:
> 1. **MobileNetV3-Small baseline** — train thường, KHÔNG teacher (mốc đối chứng).
> 2. **Teacher ViT-B/16** — thầy.
> 3. **CAKD (KD)** — MobileNetV3-Small học từ ViT (distillation). Cùng backbone với mốc 1, chỉ khác có/không KD.
>
> Cuối cùng có **1 ô so sánh** cả 3 (bảng + biểu đồ cột accuracy / F1). Student & teacher đều xuất 4 đầu ra
> cùng shape (attn 196×196, feat 196×768, token 768) nên mọi loss CAKD khớp y nguyên.
>
> Chuẩn bị: New Notebook → **Settings → Accelerator → GPU T4/P100** → Add Input: thêm dataset `15k-image-trash`.

---

# ⚙️ SETUP (chạy 1 lần)

## Ô 1 — Xác nhận đường dẫn dataset
```bash
!ls /kaggle/input
!ls /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image
```
→ Lệnh thứ 2 phải in ra `glass  paper  plastic`. Nếu path khác, sửa lại `--data-path`/`--src` ở các ô dưới.

## Ô 2 — Clone code (nhánh mobilenetv3-small) + cài einops
```bash
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git fetch origin && git checkout mobilenetv3-small && git pull) \
  || git clone -b mobilenetv3-small https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops
```

## Ô 3 — Reorg ảnh → ImageFolder (cho train CAKD + teacher)
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
> Mọi ô "metric" bên dưới đều chấm trên **cùng tập test này** (`data_test/val`) → so sánh công bằng.

## Ô 5 — Verify model dựng đúng
```bash
%cd /kaggle/working/repo/CAKD
!python -c "import torch; from models.mobilenet_cakd import mobilenetv3_small_cakd; from models.vit_cakd import build_teacher; x=torch.randn(2,3,224,224).cuda(); s=mobilenetv3_small_cakd(num_classes=3).cuda().eval(); o=s(x); t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); print('student', tuple(o[0].shape), '| teacher', tuple(to[0].shape), '| params(M)', round(sum(p.numel() for p in s.parameters())/1e6,2), '| OK')"
```
→ Phải in `student (2, 3) | teacher (2, 3) | params(M) 2.58 | OK`.

> 💡 **Muốn chạy nhanh** ở mọi ô train: thêm `--data-fraction 0.25` (chỉ 1/4 tập train, chia đều theo lớp)
> và/hoặc giảm `--epochs`. Bỏ đi khi chạy chính thức.

---

# 1️⃣ MOBILENETV3-SMALL BASELINE (không KD — mốc đối chứng)

## Ô 6 — Train baseline
```bash
!python /kaggle/working/repo/kaggle_train_mobilenetv3_baseline.py \
  --data-path /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --epochs 30 --batch-size 32 --lr 0.01 \
  --output-dir /kaggle/working/baseline_mobilenetv3
```
→ Sinh `baseline_mobilenetv3/mobilenetv3_baseline_best.pth` + `history_mobilenetv3_baseline.json`. In `TEST acc@1`.

## Ô 7 — Biểu đồ quá trình train baseline
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/baseline_mobilenetv3/history_mobilenetv3_baseline.json \
  --out /kaggle/working/plot_baseline.png --title "MobileNetV3 baseline"
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

# 2️⃣ TEACHER ViT-B/16

## Ô 9 — Fine-tune teacher ViT → 3 lớp
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 20 --lr 2e-4 --wd 0.05 --label-smoothing 0.1 \
  --amp --output-dir /kaggle/working
```
→ Sinh `/kaggle/working/teacher_3cls.pth` + `history_teacher.json`. **Cần teacher tốt trước khi sang mốc 3.**

## Ô 10 — Biểu đồ quá trình train teacher
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/history_teacher.json \
  --out /kaggle/working/plot_teacher.png --title "Teacher ViT-B/16"
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

# 3️⃣ CAKD — KD: ViT → MobileNetV3-Small

## Ô 12 — Distill (train student CAKD)
> Bám công thức gốc (auto-augment ta_wide, random-erase 0.1, mixup 0.2, label-smoothing 0.1, wd 2e-5, norm-wd 0,
> ra-sampler reps 4, model-ema, 120 epoch, λ khởi động ep25 ramp 50). **1 GPU:** `--nproc_per_node=1`, `--lr 0.0125`.
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
→ Log in `pca_loss / gl_loss / cls_loss / gan_loss`. Sinh `results/checkpoint.pth` + `results/history_cakd.json`.

> ⚠️ 120 epoch/1 GPU khá lâu. Muốn nhanh: thêm `--data-fraction 0.25` và/hoặc
> `--epochs 40 --distill-start 8 --distill-ramp 16 --lr 0.01`. Backbone nhẹ nên có thể hạ `--lr 0.008` nếu loss khó xuống.

## Ô 13 — Biểu đồ quá trình train CAKD
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/results/history_cakd.json \
  --out /kaggle/working/plot_cakd.png --title "CAKD student (MobileNetV3)"
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
- **Đọc kết quả:** nếu **MobileNet (CAKD) > MobileNet (baseline)** → KD ăn thua. So với **ViT (teacher)** để thấy
  student nén được bao nhiêu % kiến thức của thầy dù nhẹ hơn ~33×.

---

## Ghi chú kỹ thuật
- **3 mốc dùng chung tập test** `data_test/val` + cùng preprocessing trong `eval_metrics.py` → so sánh công bằng.
- **Student** `mobilenetv3_small_cakd` (`CAKD/models/mobilenet_cakd.py`, ~2.6M): tách backbone tại tầng 14×14
  (48 kênh/196 token) cắm 2 đầu distill `pca_proj`/`gl_proj`; `features[9:]`+classifier lo phân loại. Đổi arch
  qua `--student-arch resnet18|resnet50` nếu muốn bản nặng (nhớ đổi cả ô train, eval, biểu đồ cho khớp).
- **Baseline** dùng `torchvision.mobilenet_v3_small` thường (output logits), đánh giá qua `eval_metrics --model baseline`.
- **λ (lịch distill):** `min(max(epoch-25,0)/50, 0.2)` qua `--distill-start 25 --distill-ramp 50`.

## Sự cố thường gặp
- **Lỗi shape ở `gl_loss`** → teacher chưa 3 lớp: kiểm tra `--teacher-weights` trỏ đúng `teacher_3cls.pth`.
- **`CUDA out of memory`** → giảm `--batch-size` 16/8, giữ `--amp`.
- **`load_state_dict` lỗi khi eval** → dùng SAI `--model`/`--student-arch`: baseline↔`--model baseline`,
  student CAKD↔`--model student --student-arch mobilenetv3_small`, teacher↔`--model teacher`.
- **`... modified by an inplace operation`** → chạy code cũ: `!cd /kaggle/working/repo && git pull`.
- **Chạy train mà "không thấy log gì"** → KHÔNG treo. Lần đầu mất vài phút để quét ~15K ảnh + tải backbone
  pretrained (MobileNetV3-Small ~10MB) + dựng sampler. Dùng `PYTHONUNBUFFERED=1` (đã có sẵn) để log hiện ngay.

## Chạy thử nhanh trước khi chạy full
Chạy mỗi mốc 2 epoch + 1/4 data để chắc pipeline thông, rồi mới chạy full:
```bash
# baseline thử
!python /kaggle/working/repo/kaggle_train_mobilenetv3_baseline.py --epochs 2 --data-fraction 0.25 \
  --output-dir /kaggle/working/baseline_mobilenetv3
# teacher thử
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if --batch-size 16 --epochs 2 --amp --output-dir /kaggle/working
# CAKD thử
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch mobilenetv3_small --student-pretrained --workers 4 --batch-size 16 --lr 0.0125 --epochs 2 \
  --data-fraction 0.25 --distill-start 0 --distill-ramp 2 --amp --output-dir /kaggle/working/results
```
