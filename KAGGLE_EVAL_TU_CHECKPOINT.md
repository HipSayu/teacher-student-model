# Đánh giá từ checkpoint có sẵn trên Kaggle (teacher + KD) — copy từng ô

> Dùng khi bạn **đã có file `.pth`** (upload/đính kèm trên Kaggle) và chỉ muốn **chấm trên tập test** +
> ra **metric đầy đủ (accuracy, precision/recall/F1, macro/weighted) + ma trận nhầm lẫn**, KHÔNG train lại.
>
> Đường dẫn dùng trong guide (theo dataset của bạn — sửa nếu khác):
> - Ảnh gốc 3 lớp: `/kaggle/input/datasets/hieppn/15k-data/15K_Image` (cấu trúc `<class>/<split>/images/`)
> - Teacher: `/kaggle/input/notebooks/hieppn/final-teacher/teacher_3cls_best.pth`
> - KD student: `/kaggle/input/notebooks/hieppn/final-teacher/results/checkpoint.pth`
>
> Không cần sửa code — `eval_metrics.py` trong repo lo hết.

## Chuẩn bị
1. **New Notebook** → Settings → **GPU T4/P100**.
2. **Add Input**: dataset ảnh `15k-data` + notebook/dataset chứa các file `.pth`.

---

## Ô 1 — Clone code + dựng tập TEST
```bash
# clone code (chứa eval_metrics.py + tools)
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git fetch origin && git checkout mobilenetv3-small && git pull) \
  || git clone -b mobilenetv3-small https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops

# dựng tập TEST dạng ImageFolder (data_test/val) từ dataset gốc 3 lớp
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/datasets/hieppn/15k-data/15K_Image \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
```
> Log phải in `test : tong=1500 (glass=500, paper=500, plastic=500)`.
> Nếu in **0 ảnh** → thư mục split KHÔNG có `images/` bên trong. Kiểm tra bằng
> `!ls /kaggle/input/datasets/hieppn/15k-data/15K_Image/glass/test` rồi báo mình để chỉnh.

## Ô 2 — Kiểm tra 2 file checkpoint có đúng đường dẫn không
```bash
!ls -la /kaggle/input/notebooks/hieppn/final-teacher/teacher_3cls_best.pth
!ls -la /kaggle/input/notebooks/hieppn/final-teacher/results/checkpoint.pth
```
→ Cả 2 phải hiện ra (không `No such file`). Nếu sai path, `!ls /kaggle/input/notebooks/hieppn/final-teacher` để dò lại.

---

## Ô 3 — Eval TEACHER trên test (từ `teacher_3cls_best.pth`)
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py --model teacher \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/input/notebooks/hieppn/final-teacher/teacher_3cls_best.pth \
  --out-dir /kaggle/working/results

from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_teacher.png'))
```
→ In accuracy + precision/recall/F1 từng lớp + macro/weighted; lưu `results/metrics_teacher.json`
  + `results/confusion_matrix_teacher.png`.

## Ô 4 — Eval KD student trên test (từ `results/checkpoint.pth`)
```python
%cd /kaggle/working/repo/CAKD
# ⚠️ --student-arch PHẢI khớp kiến trúc lúc train KD (resnet18 / resnet50 / mobilenetv3_small)
!python eval_metrics.py --model student --student-arch resnet18 --weights ema \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/input/notebooks/hieppn/final-teacher/results/checkpoint.pth \
  --out-dir /kaggle/working/results

from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_student_resnet18.png'))
```
→ Lưu `results/metrics_student_resnet18.json` + confusion matrix.
> Nếu báo lỗi `load_state_dict` (size mismatch) → checkpoint KD KHÔNG phải resnet18. Đổi `--student-arch`
> thành `resnet50` hoặc `mobilenetv3_small` cho khớp, và tên file confusion sẽ đổi theo (vd
> `confusion_matrix_student_mobilenetv3_small.png`).

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
> Nếu ở Ô 4 bạn đổi arch, sửa tên file `metrics_student_<arch>.json` cho khớp.

---

## Lưu ý quan trọng
- **`--student-arch` phải KHỚP** kiến trúc lúc train KD, nếu sai sẽ lỗi `load_state_dict`. Đang để `resnet18`.
- **`--weights ema`** (mặc định) khớp số "best" khi train KD có `--model-ema`. Nếu checkpoint không có
  `model_ema`, script tự rớt về trọng số `model` — vẫn chạy.
- **Model thường (không CAKD)** — nếu file `.pth` là baseline train riêng (torchvision thuần, output logits),
  dùng `--model baseline --student-arch <arch>` thay vì `--model student`.
- **Chỉ có metric + confusion trên test.** Từ file `.pth` KHÔNG dựng lại được đường cong train — muốn vẽ
  đường cong phải có kèm `history_*.json` rồi chạy `tools/plot_training.py --history <file>`.
- Cả teacher lẫn KD đều chấm trên **cùng tập test** `data_test/val` → so sánh công bằng.

## Sự cố thường gặp
- **reorg in 0 ảnh** → thư mục `<class>/<split>` không có `images/`; kiểm tra `!ls .../glass/test`.
- **`load_state_dict` size mismatch** → sai `--student-arch` (Ô 4) hoặc sai `--model`.
- **`FileNotFoundError` cho `.pth`** → path `--checkpoint` sai; chạy lại Ô 2 để dò path đúng.
- **`Không thấy .../val`** → chưa chạy Ô 1 dựng `data_test/val`.
