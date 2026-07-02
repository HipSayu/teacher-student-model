# Chặng 1 — `CAKD/new_utils.py`

> File này chứa nhiều tiện ích (logger, metric, distributed…) lấy từ torchvision references.
> CAKD **chỉ tự viết 2 lớp**: `GANLoss` và `NLayerDiscriminator`. Đọc kỹ 2 lớp đó, còn lại ⚪ LƯỚT.

---

## 🟢 `GANLoss` — dòng 20–79

Đóng gói hàm loss của GAN, tự tạo tensor nhãn (toàn 1 = "thật", toàn 0 = "giả") đúng kích thước đầu ra của Discriminator.

```python
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))  # nhãn "thật" = 1.0, lưu như buffer (theo model .to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))  # nhãn "giả"  = 0.0
        self.gan_mode = gan_mode                    # CAKD dùng 'vanilla'
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()                # LSGAN: bình phương sai
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()      # ← CAKD DÙNG CÁI NÀY: BCE + sigmoid gộp sẵn (ổn định số học)
        elif gan_mode in ['wgangp']:
            self.loss = None                        # WGAN-GP tính tay, không cần hàm loss
        else:
            raise NotImplementedError(...)
```
> **Vì sao `BCEWithLogitsLoss`?** Vì Discriminator KHÔNG có sigmoid ở cuối (xem lớp dưới). BCEWithLogits tự thêm sigmoid bên trong → tránh tràn số. Nhớ dòng comment trong code: *"Do not use sigmoid as the last layer of Discriminator."*

```python
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label         # chọn 1.0
        else:
            target_tensor = self.fake_label         # chọn 0.0
        return target_tensor.expand_as(prediction)  # "phình" scalar thành tensor CÙNG SHAPE với prediction
```
> Mấu chốt: `expand_as` giúp mày gọi `gan_criterion(pred, True)` mà không cần tự tạo tensor nhãn — nó tự đúng shape.

```python
    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)  # tạo nhãn
            loss = self.loss(prediction, target_tensor)                         # BCE(pred, nhãn)
        elif self.gan_mode == 'wgangp':
            loss = -prediction.mean() if target_is_real else prediction.mean()  # WGAN (không dùng ở đây)
        return loss
```
> Cách đọc câu lệnh trong train:
> - `gan_criterion(pred_real, True)`  → ép `pred_real → 1` (D nói "thật").
> - `gan_criterion(pred_fake, False)` → ép `pred_fake → 0` (D nói "giả").
> - `gan_criterion(pred_fake, True)`  → ép `pred_fake → 1` (HỌC SINH muốn lừa D). ← mẹo non-saturating.

---

## 🟢 `NLayerDiscriminator` — dòng 81–126 (PatchGAN)

"Trọng tài" nhìn attention map (coi như ảnh 1 kênh) và chấm điểm thật/giả. Kiến trúc = xếp chồng Conv-Norm-LeakyReLU.

```python
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # ── xử lý bias: nếu norm là InstanceNorm thì cần bias, BatchNorm thì không (đã có affine) ──
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4; padw = 1                             # kernel 4×4, padding 1 (chuẩn PatchGAN)
        # ── Lớp đầu: Conv giảm 1/2 kích thước, KHÔNG norm, có LeakyReLU ──
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1; nf_mult_prev = 1
        # ── Các lớp giữa: tăng dần số kênh (×2 mỗi lớp, trần ×8), mỗi lớp giảm 1/2 kích thước ──
        for n in range(1, n_layers):                # n_layers=3 → chạy n=1,2
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)                 # 2, 4 (trần 8)
            sequence += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        # ── Lớp áp chót: stride=1 (KHÔNG giảm kích thước nữa), vẫn tăng kênh ──
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf*nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # ── Lớp cuối: gộp về 1 kênh = "bản đồ điểm" thật/giả (chưa qua sigmoid) ──
        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        sequence += [nn.AdaptiveAvgPool2d(1)]        # ép bản đồ điểm về 1 số duy nhất / ảnh
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)                     # (B,1,H,W) → (B,1,1,1) điểm số
```
> **Điểm cần nhớ:**
> - Trong train khởi tạo `NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3)` → **input 1 kênh** vì attention map là "ảnh xám" 196×196.
> - Đầu ra là **điểm số thô** (logits), chưa sigmoid → khớp với `BCEWithLogitsLoss` ở `GANLoss`.
> - "PatchGAN" = chấm điểm theo từng vùng (patch) thay vì cả ảnh, nhưng cuối cùng `AdaptiveAvgPool2d(1)` gộp lại 1 số.

---

## ⚪ Phần còn lại của file (LƯỚT)

Đọc để biết nó có, **không cần soi từng dòng** — đều là tiện ích chuẩn từ torchvision references:

| Thành phần | Vai trò (đủ để biết) |
|------------|----------------------|
| `SmoothedValue`, `MetricLogger` | in log `loss/acc/img_s` theo cửa sổ trượt trong lúc train |
| `accuracy(output, target, topk)` | tính Acc@1 / Acc@5 |
| `ClassificationPresetTrain/Eval` | bộ transform (resize/crop/normalize/augment) cho ảnh |
| `RASampler` | Repeated Augmentation sampler (bật bởi `--ra-sampler`) |
| `ExponentialMovingAverage` | bản EMA của model (`--model-ema`) |
| `set_weight_decay` | tách nhóm tham số áp weight-decay khác nhau |
| `init_distributed_mode`, `reduce_across_processes`, `save_on_master`, `mkdir` | tiện ích chạy đa GPU |

---

## ✅ Kiểm tra hiểu trước khi sang chặng 2
1. `gan_criterion(pred_fake, True)` ép điểm số của D về giá trị nào, và tại sao học sinh lại muốn thế?
2. Vì sao Discriminator không có sigmoid ở lớp cuối?
3. `input_nc=1` nghĩa là gì trong bối cảnh CAKD?
