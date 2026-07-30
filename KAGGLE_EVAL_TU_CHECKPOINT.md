# Đánh giá từ checkpoint có sẵn trên Kaggle (teacher + KD) — copy từng ô

> Dùng khi bạn **đã có file `.pth`** (upload lại lên Kaggle) và chỉ muốn **chấm trên tập test** + ra
> **metric đầy đủ (accuracy, precision/recall/F1, macro/weighted) + ma trận nhầm lẫn**, KHÔNG train lại.
>
> Áp dụng cho:
> - **Teacher ViT** — file `teacher_3cls_best.pth` (hoặc `teacher_3cls.pth`).
> - **KD student (CAKD)** — file `checkpoint.pth`.
>
> Không cần sửa code — `eval_metrics.py` trong repo lo hết.

## Chuẩn bị
1. **New Notebook** → Settings → **GPU T4/P100**.
2. **Add Input**:
   - Dataset ảnh gốc 3 lớp `15k-image-trash` (để lấy tập test).
   - Dataset chứa các file checkpoint `.pth` bạn vừa upload.

---

## Ô 1 — Clone code + dựng tập TEST
```bash
# clone code (chứa eval_metrics.py + tools)
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git fetch origin && git checkout mobilenetv3-small && git pull) \
  || git clone -b mobilenetv3-small https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops

# dựng tập TEST dạng ImageFolder (data_test/val) từ dataset gốc 3 lớp
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
```

## Ô 2 — Tìm đúng đường dẫn file checkpoint đã upload
```bash
!ls /kaggle/input
!ls /kaggle/input/<ten-dataset-checkpoint>       # tìm teacher_3cls_best.pth, checkpoint.pth
```
→ Ghi nhớ đường dẫn đầy đủ tới từng file `.pth` để điền vào các ô dưới.

---

## Ô 3 — Eval TEACHER trên test (từ `teacher_3cls_best.pth`)
```python
%cd /kaggle/working/repo/CAKD
# ⚠️ Sửa --checkpoint cho đúng nơi bạn upload
!python eval_metrics.py --model teacher \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/input/<ten-dataset-checkpoint>/teacher_3cls_best.pth \
  --out-dir /kaggle/working/results

from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_teacher.png'))
```
→ In accuracy + precision/recall/F1 từng lớp + macro/weighted; lưu `results/metrics_teacher.json`
  + `results/confusion_matrix_teacher.png`.

## Ô 4 — Eval KD student trên test (từ `checkpoint.pth`)
```python
%cd /kaggle/working/repo/CAKD
# ⚠️ --student-arch PHẢI khớp lúc train KD (resnet18 / resnet50 / mobilenetv3_small)
# ⚠️ Sửa --checkpoint cho đúng file KD bạn upload
!python eval_metrics.py --model student --student-arch resnet18 --weights ema \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/input/<ten-dataset-checkpoint>/checkpoint.pth \
  --out-dir /kaggle/working/results

from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_student_resnet18.png'))
```
→ Lưu `results/metrics_student_resnet18.json` + confusion matrix.
  (Nếu KD là arch khác thì đổi `--student-arch` và tên file confusion matrix theo đó,
  vd `confusion_matrix_student_mobilenetv3_small.png`.)

## Ô 5 — So sánh teacher vs KD (bảng + biểu đồ cột)
```python
!python /kaggle/working/repo/tools/compare_metrics.py \
  --metrics /kaggle/working/results/metrics_teacher.json \
            /kaggle/working/results/metrics_student_resnet18.json \
  --out /kaggle/working/compare.png
from IPython.display import Image, display
display(Image('/kaggle/working/compare.png'))
```
→ In bảng accuracy/macro-P/R/F1 + F1 từng lớp, và biểu đồ cột so sánh.
  (Có `metrics_baseline_*.json` thì thêm vào `--metrics` để so cả 3.)

---

## Lưu ý quan trọng
- **`--student-arch` phải KHỚP** kiến trúc lúc train KD, nếu sai sẽ lỗi `load_state_dict`.
  Đang để mặc định `resnet18` — đổi nếu checkpoint của bạn là `resnet50` hoặc `mobilenetv3_small`.
- **`--weights ema`** (mặc định) khớp số "best" khi train có `--model-ema`. Nếu checkpoint không có
  `model_ema`, script tự rớt về trọng số `model` — vẫn chạy.
- **Model thường (không CAKD)** — nếu file `.pth` là baseline train riêng (torchvision thuần, output logits),
  dùng `--model baseline --student-arch <arch>` thay vì `--model student`.
- **Chỉ có metric + confusion trên test.** Từ file `.pth` KHÔNG dựng lại được đường cong train (loss/acc theo
  epoch) — muốn vẽ đường cong phải upload kèm `history_teacher.json` / `history_cakd.json` rồi chạy
  `tools/plot_training.py --history <file>`.
- Cả teacher lẫn KD đều chấm trên **cùng tập test** `data_test/val` → so sánh công bằng.

## Sự cố thường gặp
- **`load_state_dict` lỗi** → sai `--student-arch` hoặc sai `--model`. Teacher↔`--model teacher`;
  KD CAKD↔`--model student --student-arch <arch>`; baseline thường↔`--model baseline --student-arch <arch>`.
- **`FileNotFoundError` cho `.pth`** → đường dẫn `--checkpoint` sai; chạy lại Ô 2 để lấy path đúng.
- **`Không thấy .../val`** → chưa chạy Ô 1 dựng `data_test/val`.
