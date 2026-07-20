# Chạy CAKD 3 lớp (glass / paper / plastic) trên Kaggle — copy từng ô

> **Bản port torch 2.x — chạy native, KHÔNG cần ghim torch 1.12, KHÔNG cần chép đè file torch.**
>
> **Student = ResNet-18** (mặc định, ~15M tham số — nhẹ hơn ResNet-50 ~42M). Chọn kiến trúc qua
> cờ `--student-arch resnet18|resnet50`. Teacher vẫn là ViT-B/16. Mọi loss (pca/gl/gan) khớp y
> nguyên vì cả hai arch đều xuất 4 đầu ra cùng shape (attn 196×196, feat 196×768, token 768).
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
![ -d /kaggle/working/repo ] && (cd /kaggle/working/repo && git fetch origin && git checkout resnet18 && git pull) \
  || git clone -b resnet18 https://github.com/HipSayu/teacher-student-model.git /kaggle/working/repo
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
!python -c "import torch; from models.resnet_cakd import resnet18_cakd; from models.vit_cakd import build_teacher; x=torch.randn(2,3,224,224).cuda(); s=resnet18_cakd(num_classes=3).cuda().eval(); o=s(x); t=build_teacher(3,pretrained=False).cuda().eval(); to=t(x); print('student', tuple(o[0].shape), '| teacher', tuple(to[0].shape), '| params(M)', round(sum(p.numel() for p in s.parameters())/1e6,1), '| OK')"
```
→ Phải in `student (2, 3) | teacher (2, 3) | params(M) 15.0 | OK`.

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

## Ô 5 — GĐ2: CAKD distill ViT(3 lớp) → ResNet-18
> Bám sát công thức gốc `experiments/run_cakd.sh` + bài báo (auto-augment ta_wide, random-erase 0.1,
> mixup 0.2, label-smoothing 0.1, weight-decay 2e-5, norm-wd 0, ra-sampler reps 4, model-ema, 120 epoch,
> lịch λ khởi động ở epoch 25 ramp 50). **Chỉ đổi cho 1 GPU:** `--nproc_per_node=1` và
> `--lr 0.0125` (= 0.1 × 32/256 theo quy tắc linear-scaling, vì batch 32 thay vì 256=8×32).
> **Thêm cho bài toán 3 lớp:** `--teacher-weights`, `--student-pretrained`.
> **Student ResNet-18:** thêm `--student-arch resnet18` (mặc định; đổi `resnet50` nếu muốn bản nặng).
```bash
%cd /kaggle/working/repo/CAKD
!PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch resnet18 --student-pretrained --workers 4 \
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
  --src /kaggle/input/datasets/triuquct/15k-image-trash/15K_Image \
  --dst /kaggle/working/data_test --splits test
!mkdir -p /kaggle/working/data_test/val && cp -r /kaggle/working/data_test/test/* /kaggle/working/data_test/val/
%cd /kaggle/working/repo/CAKD
!torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_test \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-arch resnet18 \
  --test-only --resume /kaggle/working/results/checkpoint.pth --batch-size 32 --workers 4
```
> Chỉ nhìn **Acc@1** (với 3 lớp thì "Acc@5" thực chất là Acc@3, không quan trọng).

## Ô 7 — Demo suy luận 1 ảnh
```python
import torch, os
from PIL import Image
from models.resnet_cakd import resnet18_cakd
from new_utils import ClassificationPresetEval

classes = ["glass", "paper", "plastic"]        # đúng thứ tự alphabet của ImageFolder
model = resnet18_cakd(num_classes=3).cuda().eval()   # phải KHỚP --student-arch lúc train
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

## Ô 9 — Metric chi tiết + ma trận nhầm lẫn (precision / recall / f1)
> File `history_*.json` **chỉ có** accuracy tổng mỗi epoch — KHÔNG đủ để dựng ma trận nhầm lẫn.
> Ô này chạy **1 lượt inference từ checkpoint** (KHÔNG train lại) trên tập test để tính đầy đủ:
> accuracy, precision/recall/f1 từng lớp, macro/weighted avg, và ma trận nhầm lẫn.
> Cần chạy **Ô 6 trước** (nó tạo `/kaggle/working/data_test/val`). `scikit-learn` đã có sẵn trên Kaggle.
```python
%cd /kaggle/working/repo/CAKD
!python eval_metrics.py \
  --data-path /kaggle/working/data_test \
  --checkpoint /kaggle/working/results/checkpoint.pth \
  --student-arch resnet18 --weights ema \
  --out-dir /kaggle/working/results

from IPython.display import Image, display
display(Image('/kaggle/working/results/confusion_matrix.png'))
```
- In ra bảng `precision / recall / f1 / support` từng lớp + accuracy tổng + ma trận nhầm lẫn (dạng số).
- Lưu `results/metrics.json` (mọi số liệu) + `results/confusion_matrix.png` (2 ma trận: counts & chuẩn hoá theo hàng = recall).
- `--weights ema` khớp đúng số **best** lúc train (vì có `--model-ema`); đổi `--weights model` nếu muốn trọng số thường.

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
  --student-arch resnet18 --student-pretrained --workers 4 --batch-size 16 --lr 0.0125 --epochs 2 \
  --distill-start 0 --distill-ramp 2 --amp --output-dir /kaggle/working/results
```

## Ghi chú kỹ thuật (port torch 2.x — khác gì bản gốc)
- **Teacher** dùng thẳng `torchvision.models.vit_b_16(weights=...)` thật; logits **trùng khít** ViT gốc,
  tự moi attention qk/vv 2 lớp cuối (khớp đúng bản sửa `functional.py` cũ) — không cần sửa torch.
- **Student** `resnet18_cakd` / `resnet50_cakd` là module local (`CAKD/models/resnet_cakd.py`),
  device-agnostic, chọn qua `--student-arch` (mặc định `resnet18`). ResNet-18 kênh sau layer3 = 256
  (R50 là 1024) nhưng `pca_proj`/`gl_proj` tự bám `self.inplanes` nên khớp teacher không cần chỉnh tay.
  ⚠️ Lúc test (Ô 6) / demo (Ô 7) / deploy phải dùng **đúng arch** đã train, nếu không load_state_dict lỗi.
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
