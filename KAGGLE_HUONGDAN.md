# Chạy CAKD 3 lớp (glass / paper / plastic) trên Kaggle — copy từng ô

> **Bản port torch 2.x — chạy native, KHÔNG cần ghim torch 1.12, KHÔNG cần chép đè file torch.**
>
> **Student = MobileNetV3-Small** (nhánh này, mặc định, ~2.6M tham số — SIÊU NHẸ, nhẹ ~6× so với
> ResNet-18 ~15M, ~16× so với ResNet-50 ~42M). Vẫn có pretrained ImageNet. Chọn kiến trúc qua cờ
> `--student-arch mobilenetv3_small|resnet18|resnet50`. Teacher vẫn là ViT-B/16. Mọi loss (pca/gl/gan)
> khớp y nguyên vì mọi arch đều xuất 4 đầu ra cùng shape (attn 196×196, feat 196×768, token 768) —
> student cắm 2 đầu distill tại tầng feature 14×14.
>
> Chuẩn bị: New Notebook → **Settings → Accelerator → GPU T4/P100** → Add Input: thêm dataset
> `15k-image-trash`. Dataset ảnh dạng `class/split/images` (glass/paper/plastic → train/val/test).

---

## Ô 0 — Xác nhận đường dẫn dataset
```bash
!ls /kaggle/input
!ls /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image
```
→ Lệnh thứ 2 phải in ra `glass  paper  plastic`. Nếu path khác, thay lại `--src` ở Ô 2 & Ô 6.

## Ô 1 — Clone code (hoặc cập nhật nếu đã clone) + cài einops
```bash
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git fetch origin && git checkout mobilenetv3-small && git pull) \
  || git clone -b mobilenetv3-small https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops
```
> Ô này chạy lại nhiều lần được: chưa có thì clone, có rồi thì `git pull` lấy bản mới nhất.

## Ô 2 — Reorg ảnh → ImageFolder (bỏ labels/, chỉ lấy ảnh)
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_if \
  --classes glass paper plastic --splits train val test
```

## Ô 3 — Verify model dựng đúng (thay bước "chép file" cũ)
```bash
%cd /kaggle/working/repo/CAKD
!python -c "import torch; from models.mobilenet_cakd import mobilenetv3_small_cakd; from models.vit_cakd import build_teacher; x=torch.randn(2,3,224,224).cuda(); s=mobilenetv3_small_cakd(num_classes=3).cuda().eval(); o=s(x); t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); print('student', tuple(o[0].shape), '| teacher', tuple(to[0].shape), '| params(M)', round(sum(p.numel() for p in s.parameters())/1e6,2), '| OK')"
```
→ Phải in `student (2, 3) | teacher (2, 3) | params(M) 2.58 | OK`.

## Ô 4 — GĐ1: fine-tune teacher ViT-B/16 → 3 lớp
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 20 --lr 2e-4 --wd 0.05 --label-smoothing 0.1 \
  --amp --output-dir /kaggle/working
```
Kết thúc: có `/kaggle/working/teacher_3cls.pth` + in `best acc@1=...`. Kiểm tra teacher đạt accuracy
tốt **trước khi** sang Ô 5.

## Ô 5 — GĐ2: CAKD distill ViT(3 lớp) → MobileNetV3-Small
> Bám sát công thức gốc `experiments/run_cakd.sh` + bài báo (auto-augment ta_wide, random-erase 0.1,
> mixup 0.2, label-smoothing 0.1, weight-decay 2e-5, norm-wd 0, ra-sampler reps 4, model-ema, 120 epoch,
> lịch λ khởi động ở epoch 25 ramp 50). **Chỉ đổi cho 1 GPU:** `--nproc_per_node=1` và
> `--lr 0.0125` (= 0.1 × 32/256 theo quy tắc linear-scaling, vì batch 32 thay vì 256=8×32).
> **Thêm cho bài toán 3 lớp:** `--teacher-weights`, `--student-pretrained`.
> **Student MobileNetV3-Small:** `--student-arch mobilenetv3_small` (mặc định nhánh này; đổi `resnet18`
> / `resnet50` nếu muốn bản nặng hơn). Backbone nhẹ nên nếu loss khó xuống có thể hạ `--lr 0.008`.
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
Log phải in `pca_loss / gl_loss / cls_loss / gan_loss` (KHÔNG lỗi shape). Sinh `results/checkpoint.pth`.

> ⚠️ 120 epoch trên 15K ảnh / 1 GPU khá lâu (nhiều giờ). Muốn nhanh:
> - **Chạy 1/4 dữ liệu:** thêm `--data-fraction 0.25` (chỉ lấy 1/4 tập TRAIN, **chia đều theo lớp**;
>   val giữ nguyên để accuracy vẫn trung thực). Mỗi epoch nhanh ~4×. Dùng `0.5` cho 1/2, v.v.
> - **Rút ngắn lịch:** giảm `--epochs 40 --distill-start 8 --distill-ramp 16 --lr 0.01`
>   (vẫn giữ đúng phương pháp, chỉ rút ngắn số epoch).
>
> Ví dụ chạy nhẹ (1/4 data + 40 epoch) — thêm 2 cờ vào lệnh Ô 5:
> ```bash
> !PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
>   --data-path /kaggle/working/data_if --teacher-weights /kaggle/working/teacher_3cls.pth \
>   --student-arch mobilenetv3_small --student-pretrained --workers 4 \
>   --data-fraction 0.25 --batch-size 32 --lr 0.01 \
>   --auto-augment ta_wide --epochs 40 --random-erase 0.1 --mixup-alpha 0.2 \
>   --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 \
>   --distill-start 8 --distill-ramp 16 --model-ema --ra-sampler --ra-reps 4 --amp \
>   --output-dir /kaggle/working/results
> ```

## Ô 6 — Đánh giá trên tập test
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
%cd /kaggle/working/repo/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_test \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch mobilenetv3_small \
  --test-only --resume /kaggle/working/results/checkpoint.pth --batch-size 32 --workers 4
```
> Chỉ nhìn **Acc@1** (với 3 lớp thì "Acc@5" thực chất là Acc@3, không quan trọng).

## Ô 7 — Demo suy luận 1 ảnh
```python
import torch, os
from PIL import Image
from models.mobilenet_cakd import mobilenetv3_small_cakd
from new_utils import ClassificationPresetEval

classes = ["glass", "paper", "plastic"]        # đúng thứ tự alphabet của ImageFolder
model = mobilenetv3_small_cakd(num_classes=3).cuda().eval()   # phải KHỚP --student-arch lúc train
ck = torch.load("/kaggle/working/results/checkpoint.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
tf = ClassificationPresetEval(crop_size=224, resize_size=224)

folder = "/kaggle/input/datasets/triuquct/15k-image-trash/15K_Image/glass/test/images"
fname = sorted(os.listdir(folder))[0]
img = tf(Image.open(os.path.join(folder, fname)).convert("RGB")).unsqueeze(0).cuda()
with torch.inference_mode():
    prob = model(img)[0].softmax(1)[0]
print("Ảnh:", fname)
for c, p in sorted(zip(classes, prob.tolist()), key=lambda x: -x[1]):
    print(f"{c:8s} {p:.3f}")
```

## Ô 8 — Vẽ biểu đồ quá trình train
> Từ bản mới, mỗi epoch các script tự ghi lịch sử ra `history_teacher.json` / `history_cakd.json`
> (loss, accuracy...). Ô này đọc chúng và vẽ biểu đồ. **Chỉ chạy được sau khi đã train bằng code mới**
> (nhớ `git pull` ở Ô 1). File history nằm cùng `--output-dir` của mỗi bước.
```python
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/history_teacher.json --out /kaggle/working/plot_teacher.png
!python /kaggle/working/repo/tools/plot_training.py \
  --history /kaggle/working/results/history_cakd.json --out /kaggle/working/plot_cakd.png

from IPython.display import Image, display
display(Image('/kaggle/working/plot_teacher.png'))   # teacher: loss + accuracy
display(Image('/kaggle/working/plot_cakd.png'))       # cakd: loss tổng + 4 loss thành phần + accuracy
```
- **Biểu đồ teacher:** 3 panel — Loss (train), Accuracy (train vs val, có best %), Learning rate (log).
- **Biểu đồ CAKD:** 4 panel — Loss tổng, các loss thành phần `cls/pca/gl/gan` (thang log), Accuracy (best %), Learning rate (log).

## Ô 9 — Metric chi tiết + ma trận nhầm lẫn (precision / recall / f1) — CẢ student LẪN teacher
> File `history_*.json` **chỉ có** accuracy tổng mỗi epoch — KHÔNG đủ để dựng ma trận nhầm lẫn.
> Ô này chạy **1 lượt inference từ checkpoint** (KHÔNG train lại) trên tập test để tính đầy đủ:
> accuracy, precision/recall/f1 từng lớp, macro/weighted avg, và ma trận nhầm lẫn.
> Cần chạy **Ô 6 trước** (nó tạo `/kaggle/working/data_test/val`). `scikit-learn` đã có sẵn trên Kaggle.
```python
%cd /kaggle/working/repo/CAKD
# (a) STUDENT MobileNetV3-Small — dùng trọng số EMA (khớp best lúc train)
!python eval_metrics.py --model student \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/results/checkpoint.pth \
  --student-arch mobilenetv3_small --weights ema \
  --out-dir /kaggle/working/results

# (b) TEACHER ViT-B/16 — cùng tập test để so sánh
!python eval_metrics.py --model teacher \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/teacher_3cls.pth \
  --out-dir /kaggle/working/results

from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix_student_mobilenetv3_small.png'))
display(Image('/kaggle/working/results/confusion_matrix_teacher.png'))
```
- Mỗi lần in bảng `precision / recall / f1 / support` từng lớp + accuracy tổng + ma trận nhầm lẫn (dạng số).
- Lưu riêng, không ghi đè nhau:
  - Student → `metrics_student_mobilenetv3_small.json` + `confusion_matrix_student_mobilenetv3_small.png`
  - Teacher → `metrics_teacher.json` + `confusion_matrix_teacher.png`
- Mỗi PNG gồm 2 ma trận: **counts** & **chuẩn hoá theo hàng** (= recall mỗi lớp).
- Student: `--weights ema` khớp số **best** lúc train (vì có `--model-ema`); đổi `--weights model` nếu muốn trọng số thường. Teacher luôn dùng trọng số `model`.

---

## Chạy thử nhanh trước khi train full
Lần đầu nên chạy 2 epoch để chắc pipeline thông (không lỗi), rồi mới chạy Ô 4/5 với epoch đầy đủ:
```bash
# teacher thử
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if --batch-size 16 --epochs 2 --amp --output-dir /kaggle/working
# distill thử
!cd /kaggle/working/repo/CAKD && PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch mobilenetv3_small --student-pretrained --workers 4 --batch-size 16 --lr 0.0125 --epochs 2 \
  --distill-start 0 --distill-ramp 2 --amp --output-dir /kaggle/working/results
```

## Ghi chú kỹ thuật (port torch 2.x — khác gì bản gốc)
- **Teacher** dùng thẳng `torchvision.models.vit_b_16(weights=...)` thật; logits **trùng khít** ViT gốc,
  tự moi attention qk/vv 2 lớp cuối (khớp đúng bản sửa `functional.py` cũ) — không cần sửa torch.
- **Student** chọn qua `--student-arch` (mặc định nhánh này = `mobilenetv3_small`):
  - `mobilenetv3_small_cakd` (`CAKD/models/mobilenet_cakd.py`) — ~2.6M, tách backbone tại tầng 14×14
    (48 kênh, 196 token) để cắm 2 đầu distill `pca_proj`/`gl_proj`, phần `features[9:]`+classifier lo phân loại.
  - `resnet18_cakd` / `resnet50_cakd` (`CAKD/models/resnet_cakd.py`) — bản nặng hơn; `pca_proj`/`gl_proj`
    tự bám `self.inplanes` (256/1024) nên khớp teacher không cần chỉnh tay.
  ⚠️ Lúc test (Ô 6) / demo (Ô 7) / eval (Ô 9) / deploy phải dùng **đúng arch** đã train, nếu không load_state_dict lỗi.
- **Thứ tự update GAN đảo lại** (student trước, discriminator sau): torch 2.x cấm `step()` xen giữa 2
  backward chung đồ thị. Đồ thị discriminator độc lập student nên **kết quả không đổi**.
- **λ (lịch distill):** giữ đúng công thức bài báo `min(max(epoch-25,0)/50, 0.2)` qua cờ
  `--distill-start 25 --distill-ramp 50`.

## Sự cố thường gặp
- **Lỗi shape ở `gl_loss` / `mse`** → teacher chưa 3 lớp: kiểm tra `--teacher-weights` trỏ đúng
  `teacher_3cls.pth` và Ô 4 đã in `Classes: ['glass','paper','plastic']`.
- **`CUDA out of memory`** → giảm `--batch-size` 16/8, giữ `--amp`.
- **`... variable ... modified by an inplace operation`** → đang chạy code cũ chưa port; chạy
  `!cd /kaggle/working/repo && git pull` để lấy bản mới nhất nhánh `mobilenetv3-small`.
- **Teacher acc thấp** → tăng `--epochs` Ô 4 (25–30) hoặc kiểm tra thống kê ảnh ở Ô 2.
- **Chạy Ô 4/5 mà "không thấy log gì"** → KHÔNG phải treo. Lần đầu nó mất vài phút để: quét ~15K
  ảnh (ImageFolder), tải trọng số backbone pretrained (MobileNetV3-Small ~10MB), dựng RASampler. Dùng
  `PYTHONUNBUFFERED=1 torchrun ...` (đã thêm sẵn) để log hiện ngay. Log đầu tiên là dòng
  `Namespace(...)` → `Loading data` → `Creating model` → `Epoch: [0] ...`. Cứ đợi tới `Epoch: [0]`.
- **`FileNotFoundError: .../data_test/train` khi test (Ô 6)** → đã sửa trong code: chạy
  `!cd /kaggle/working/repo && git pull` để lấy bản mới rồi chạy lại Ô 6.
