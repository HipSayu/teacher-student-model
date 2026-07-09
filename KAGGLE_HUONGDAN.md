# Chạy CAKD 3 lớp (glass / paper / plastic) trên Kaggle — thứ tự các ô notebook

> **Chuẩn bị:**
> - Bật GPU: Notebook → Settings → Accelerator → **GPU T4/P100**.
> - Thêm 2 dataset vào `/kaggle/input`: (1) **code** của project này, (2) **ảnh** dạng `class/split/images`.
> - Thay `<ten-dataset-code>` và `<ten-dataset-anh>` bằng tên thật trong `/kaggle/input`.

---

## Ô 1 — Cài đặt torch 1.12 + đưa code vào working
```bash
!pip install torch==1.12.0+cu113 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
!pip install einops
%cd /kaggle/working
!cp -r /kaggle/input/<ten-dataset-code>/CAKD  /kaggle/working/CAKD
!cp -r /kaggle/input/<ten-dataset-code>/tools /kaggle/working/tools
```
> Sau khi cài torch mới, **Kaggle có thể yêu cầu Restart kernel** — bấm Restart rồi chạy tiếp từ Ô 2.

## Ô 2 — Reorg ảnh → ImageFolder
```bash
!python /kaggle/working/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/<ten-dataset-anh> \
  --dst /kaggle/working/data_if \
  --classes glass paper plastic --splits train val test
```
Kiểm tra output in **thống kê số ảnh mỗi lớp** hợp lý (không có lớp = 0).

## Ô 3 — Chép 3 file "độ" + verify
```bash
%cd /kaggle/working/CAKD
!python setup_kaggle.py
```
Bắt buộc thấy dòng `[VERIFY] OK`. Nếu lỗi version → xem mục **Sự cố** cuối file.

## Ô 4 — GĐ1: fine-tune teacher ViT → 3 lớp
```bash
%cd /kaggle/working/CAKD
!torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 15 --lr 2e-4 --amp --output-dir /kaggle/working
```
Kết thúc: có `/kaggle/working/teacher_3cls.pth` + in `best acc@1=...`.
Kiểm tra teacher đạt accuracy hợp lý **trước khi** sang Ô 5.

## Ô 5 — GĐ2: CAKD distill → ResNet-50
```bash
%cd /kaggle/working/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-pretrained \
  --batch-size 32 --lr 0.01 --epochs 60 \
  --lr-warmup-epochs 5 --lr-warmup-method linear \
  --distill-start 5 --distill-ramp 20 \
  --auto-augment ta_wide --random-erase 0.1 --mixup-alpha 0.2 \
  --label-smoothing 0.1 --model-ema --amp \
  --train-crop-size 224 --val-resize-size 224 \
  --output-dir /kaggle/working/results
```
Log phải in `pca_loss / gl_loss / cls_loss / gan_loss` (KHÔNG lỗi shape). Sinh `results/checkpoint.pth`.

## Ô 6 — Đánh giá trên tập test
Tạo thư mục ImageFolder cho test rồi đổi `test` thành `val` (script eval đọc `val/`):
```bash
!python /kaggle/working/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/<ten-dataset-anh> \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
%cd /kaggle/working/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_test \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --test-only --resume /kaggle/working/results/checkpoint.pth --batch-size 32
```
> Acc@5 vô nghĩa với 3 lớp (luôn ~100%); chỉ nhìn **Acc@1**.

## Ô 7 — Demo suy luận 1 ảnh
```python
import torch, torchvision
from PIL import Image
from new_utils import ClassificationPresetEval

classes = ["glass", "paper", "plastic"]           # đúng thứ tự alphabet của ImageFolder
model = torchvision.models.resnet50_cakd(num_classes=3).cuda().eval()
ck = torch.load("/kaggle/working/results/checkpoint.pth", map_location="cpu")
model.load_state_dict(ck["model"])
tf = ClassificationPresetEval(crop_size=224, resize_size=224)

img = tf(Image.open("/kaggle/input/<ten-dataset-anh>/glass/test/images/xxx.jpg").convert("RGB"))
img = img.unsqueeze(0).cuda()
with torch.inference_mode():
    prob = model(img)[0].softmax(1)[0]
for c, p in sorted(zip(classes, prob.tolist()), key=lambda x: -x[1]):
    print(f"{c:8s} {p:.3f}")
```

---

## Sự cố thường gặp
- **`AttributeError: module 'torchvision.models' has no attribute 'resnet50_cakd'`**
  → chưa chạy Ô 3 (chưa chép file độ), hoặc chép xong chưa Restart kernel. Chạy lại Ô 3 rồi Restart.
- **Lỗi shape ở `gl_loss` / `mse`** → teacher chưa phải 3 lớp. Kiểm tra `--teacher-weights` trỏ đúng `teacher_3cls.pth` (sinh ở Ô 4).
- **Cài torch 1.12 xung đột / không tải được** → thêm `--force-reinstall`; hoặc chọn Kaggle "Environment: Pin to original" (image cũ) để dễ hạ torch.
- **`CUDA out of memory`** → giảm `--batch-size` xuống 16 hoặc 8, giữ `--amp`.
- **`RuntimeError ... expected ... cuda`** → do `GLProj` hardcode `.to('cuda')`; phải bật GPU (không chạy CPU được).
- **Teacher accuracy thấp** → tăng `--epochs` Ô 4 (20–25), hoặc kiểm tra lại nhãn/thống kê ảnh ở Ô 2.
