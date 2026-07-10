# Chạy CAKD 3 lớp (glass / paper / plastic) trên Kaggle — copy từng ô

> **Bản port torch 2.x — chạy native, KHÔNG cần ghim torch 1.12, KHÔNG cần chép đè file torch.**
>
> Chuẩn bị: New Notebook → **Settings → Accelerator → GPU T4/P100** → Add Input: thêm dataset
> `15k-image-trash`. Dataset ảnh dạng `class/split/images` (glass/paper/plastic → train/val/test).

---

## Ô 0 — Xác nhận đường dẫn dataset
```bash
!ls /kaggle/input
!ls /kaggle/input/15k-image-trash/15K_Image
```
→ Lệnh thứ 2 phải in ra `glass  paper  plastic`. Nếu path khác, thay lại `--src` ở Ô 2 & Ô 6.

## Ô 1 — Clone code (hoặc cập nhật nếu đã clone) + cài einops
```bash
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git pull) \
  || git clone -b feat/cakd-3class-kaggle https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops
```
> Ô này chạy lại nhiều lần được: chưa có thì clone, có rồi thì `git pull` lấy bản mới nhất.

## Ô 2 — Reorg ảnh → ImageFolder (bỏ labels/, chỉ lấy ảnh)
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_if \
  --classes glass paper plastic --splits train val test
```

## Ô 3 — Verify model dựng đúng (thay bước "chép file" cũ)
```bash
%cd /kaggle/working/repo/CAKD
!python -c "import torch; from models.resnet_cakd import resnet50_cakd; from models.vit_cakd import build_teacher; x=torch.randn(2,3,224,224).cuda(); s=resnet50_cakd(num_classes=3).cuda().eval(); o=s(x); t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); print('student', tuple(o[0].shape), '| teacher', tuple(to[0].shape), '| OK')"
```
→ Phải in `student (2, 3) | teacher (2, 3) | OK`.

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

## Ô 5 — GĐ2: CAKD distill ViT(3 lớp) → ResNet-50
> Bám sát công thức gốc `experiments/run_cakd.sh` + bài báo (auto-augment ta_wide, random-erase 0.1,
> mixup 0.2, label-smoothing 0.1, weight-decay 2e-5, norm-wd 0, ra-sampler reps 4, model-ema, 120 epoch,
> lịch λ khởi động ở epoch 25 ramp 50). **Chỉ đổi cho 1 GPU:** `--nproc_per_node=1` và
> `--lr 0.0125` (= 0.1 × 32/256 theo quy tắc linear-scaling, vì batch 32 thay vì 256=8×32).
> **Thêm cho bài toán 3 lớp:** `--teacher-weights`, `--student-pretrained`.
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-pretrained --workers 4 \
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

> ⚠️ 120 epoch trên 15K ảnh / 1 GPU khá lâu (nhiều giờ). Muốn nhanh: giảm `--epochs 40 --distill-start 8
> --distill-ramp 16 --lr 0.01` (vẫn giữ đúng phương pháp, chỉ rút ngắn lịch huấn luyện).

## Ô 6 — Đánh giá trên tập test
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
%cd /kaggle/working/repo/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_test \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --test-only --resume /kaggle/working/results/checkpoint.pth --batch-size 32 --workers 4
```
> Chỉ nhìn **Acc@1** (với 3 lớp thì "Acc@5" thực chất là Acc@3, không quan trọng).

## Ô 7 — Demo suy luận 1 ảnh
```python
import torch, os
from PIL import Image
from models.resnet_cakd import resnet50_cakd
from new_utils import ClassificationPresetEval

classes = ["glass", "paper", "plastic"]        # đúng thứ tự alphabet của ImageFolder
model = resnet50_cakd(num_classes=3).cuda().eval()
ck = torch.load("/kaggle/working/results/checkpoint.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
tf = ClassificationPresetEval(crop_size=224, resize_size=224)

folder = "/kaggle/input/15k-image-trash/15K_Image/glass/test/images"
fname = sorted(os.listdir(folder))[0]
img = tf(Image.open(os.path.join(folder, fname)).convert("RGB")).unsqueeze(0).cuda()
with torch.inference_mode():
    prob = model(img)[0].softmax(1)[0]
print("Ảnh:", fname)
for c, p in sorted(zip(classes, prob.tolist()), key=lambda x: -x[1]):
    print(f"{c:8s} {p:.3f}")
```

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
  --student-pretrained --workers 4 --batch-size 16 --lr 0.0125 --epochs 2 \
  --distill-start 0 --distill-ramp 2 --amp --output-dir /kaggle/working/results
```

## Ghi chú kỹ thuật (port torch 2.x — khác gì bản gốc)
- **Teacher** dùng thẳng `torchvision.models.vit_b_16(weights=...)` thật; logits **trùng khít** ViT gốc,
  tự moi attention qk/vv 2 lớp cuối (khớp đúng bản sửa `functional.py` cũ) — không cần sửa torch.
- **Student** `resnet50_cakd` là module local (`CAKD/models/resnet_cakd.py`), device-agnostic.
- **Thứ tự update GAN đảo lại** (student trước, discriminator sau): torch 2.x cấm `step()` xen giữa 2
  backward chung đồ thị. Đồ thị discriminator độc lập student nên **kết quả không đổi**.
- **λ (lịch distill):** giữ đúng công thức bài báo `min(max(epoch-25,0)/50, 0.2)` qua cờ
  `--distill-start 25 --distill-ramp 50`.

## Sự cố thường gặp
- **Lỗi shape ở `gl_loss` / `mse`** → teacher chưa 3 lớp: kiểm tra `--teacher-weights` trỏ đúng
  `teacher_3cls.pth` và Ô 4 đã in `Classes: ['glass','paper','plastic']`.
- **`CUDA out of memory`** → giảm `--batch-size` 16/8, giữ `--amp`.
- **`... variable ... modified by an inplace operation`** → đang chạy code cũ chưa port; chạy
  `!cd /kaggle/working/repo && git pull` để lấy bản mới nhất nhánh `feat/cakd-3class-kaggle`.
- **Teacher acc thấp** → tăng `--epochs` Ô 4 (25–30) hoặc kiểm tra thống kê ảnh ở Ô 2.
- **Chạy Ô 4/5 mà "không thấy log gì"** → KHÔNG phải treo. Lần đầu nó mất vài phút để: quét ~15K
  ảnh (ImageFolder), tải trọng số ResNet-50 pretrained (~100MB), dựng RASampler. Dùng
  `PYTHONUNBUFFERED=1 torchrun ...` (đã thêm sẵn) để log hiện ngay. Log đầu tiên là dòng
  `Namespace(...)` → `Loading data` → `Creating model` → `Epoch: [0] ...`. Cứ đợi tới `Epoch: [0]`.
- **`FileNotFoundError: .../data_test/train` khi test (Ô 6)** → đã sửa trong code: chạy
  `!cd /kaggle/working/repo && git pull` để lấy bản mới rồi chạy lại Ô 6.
