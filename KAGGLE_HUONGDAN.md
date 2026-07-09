# Chạy CAKD 3 lớp (glass / paper / plastic) trên Kaggle — thứ tự các ô notebook

> **Phiên bản torch 2.x (chạy native).** Code đã được port sang torch 2.x nên **KHÔNG cần** ghim
> torch 1.12, **KHÔNG cần** chép đè file torch. Dùng thẳng torch có sẵn của Kaggle.
>
> **Chuẩn bị:**
> - Bật GPU: Notebook → Settings → Accelerator → **GPU T4/P100**.
> - Add Input: thêm dataset **ảnh** của bạn (dạng `class/split/images`) vào `/kaggle/input`.
> - Thay `<ten-dataset-anh>` bằng tên thật của dataset ảnh trong `/kaggle/input`.

---

## Ô 1 — Clone code + cài einops
```bash
!git clone -b feat/cakd-3class-kaggle https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
!pip install -q einops
```
> Không cài lại torch. Code chạy với torch/torchvision mặc định của Kaggle (2.x).

## Ô 2 — Reorg ảnh → ImageFolder
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/<ten-dataset-anh> \
  --dst /kaggle/working/data_if \
  --classes glass paper plastic --splits train val test
```
Kiểm tra output in **thống kê số ảnh mỗi lớp** hợp lý (không lớp nào = 0).

## Ô 3 — Kiểm tra model dựng đúng (thay cho bước "chép file" cũ)
```bash
%cd /kaggle/working/repo/CAKD
!python -c "import torch; from models.resnet_cakd import resnet50_cakd; from models.vit_cakd import build_teacher; \
x=torch.randn(2,3,224,224).cuda(); \
s=resnet50_cakd(num_classes=3).cuda().eval(); o=s(x); \
t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); \
print('student logits', tuple(o[0].shape), '| teacher logits', tuple(to[0].shape), '| OK')"
```
→ Phải in `student logits (2, 3) | teacher logits (2, 3) | OK`.

## Ô 4 — GĐ1: fine-tune teacher ViT → 3 lớp
```bash
%cd /kaggle/working/repo/CAKD
!torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 15 --lr 2e-4 --amp --output-dir /kaggle/working
```
Kết thúc: có `/kaggle/working/teacher_3cls.pth` + in `best acc@1=...`.
Kiểm tra teacher đạt accuracy hợp lý **trước khi** sang Ô 5.

## Ô 5 — GĐ2: CAKD distill → ResNet-50
```bash
%cd /kaggle/working/repo/CAKD
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
Tạo ImageFolder cho test rồi đổi `test`→`val` (script eval đọc thư mục `val/`):
```bash
!python /kaggle/working/repo/tools/reorg_to_imagefolder.py \
  --src /kaggle/input/<ten-dataset-anh> \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
%cd /kaggle/working/repo/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_test \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --test-only --resume /kaggle/working/results/checkpoint.pth --batch-size 32
```
> Chỉ nhìn **Acc@1** (Acc@5 với 3 lớp thực chất là Acc@3, không quan trọng).

## Ô 7 — Demo suy luận 1 ảnh
```python
import torch
from PIL import Image
from models.resnet_cakd import resnet50_cakd
from new_utils import ClassificationPresetEval

classes = ["glass", "paper", "plastic"]        # đúng thứ tự alphabet của ImageFolder
model = resnet50_cakd(num_classes=3).cuda().eval()
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

## Ghi chú kỹ thuật (port torch 2.x)
- **Teacher** dùng thẳng `torchvision.models.vit_b_16(weights=...)` thật; logits **trùng khít** ViT gốc,
  và tự moi thêm attention (qk/vv) 2 lớp cuối để distill — không cần sửa `functional.py`.
- **Student** `resnet50_cakd` là module local (`CAKD/models/resnet_cakd.py`), device-agnostic.
- **Thứ tự update GAN đã đảo** (student trước, discriminator sau) vì torch 2.x cấm `step()` xen giữa 2
  backward trên đồ thị chung. Kết quả **không đổi** (đồ thị discriminator độc lập với student).
- Đã sửa `topk=(1, min(5, C))` để 3 lớp không lỗi "index k out of range".

## Sự cố thường gặp
- **Lỗi shape ở `gl_loss` / `mse`** → teacher chưa phải 3 lớp. Kiểm tra `--teacher-weights` trỏ đúng
  `teacher_3cls.pth` (sinh ở Ô 4), và Ô 4 đã in `Classes: ['glass','paper','plastic']`.
- **`CUDA out of memory`** → giảm `--batch-size` xuống 16/8, giữ `--amp`.
- **`RuntimeError ... one of the variables needed for gradient ... inplace`** → chỉ xảy ra nếu chạy
  bản code cũ (chưa port). Đảm bảo đang ở nhánh `feat/cakd-3class-kaggle` mới nhất (`git pull`).
- **Teacher accuracy thấp** → tăng `--epochs` Ô 4 (20–25), hoặc kiểm tra lại thống kê ảnh ở Ô 2.
- **Muốn chạy thử nhanh** → ở Ô 4/5 đổi `--epochs` thành `2` để test cả pipeline trước khi train full.
