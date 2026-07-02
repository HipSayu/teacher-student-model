# Phân tích từng luồng đi trong project CAKD (code ↔ bài báo)

Tài liệu này "chạy bộ" qua **mọi luồng** (data flow & execution flow) của project, từ lệnh khởi động đến lúc ra loss và suy luận, ghi rõ **shape tensor** từng bước và **ánh xạ về công thức (1)–(10)** của bài báo.

> Đọc kèm: `Phan_tich_chi_tiet_CAKD.md` (giải thích công thức) và `TaiLieu_NenTang/` (kiến thức nền).

---

## 0. Bản đồ luồng tổng (call graph)

```
experiments/run_cakd.sh
   └─ torchrun --nproc_per_node=8  dist_train_cakd.py  [các tham số]
        └─ __main__ → get_args_parser().parse_args() → main(args)
             ├─ init_distributed_mode()                         # khởi tạo đa GPU (DDP/NCCL)
             ├─ load_data() → ImageFolder + augment + Sampler   # ① LUỒNG DỮ LIỆU
             ├─ DataLoader(..., collate_fn=Mixup/CutMix)        # ① (tiếp)
             ├─ tạo model(resnet50_cakd) / teacher(vit_b_16) / discriminator
             ├─ tạo criterion(CE) / mse_criterion / gan_criterion(GANLoss)
             ├─ tạo optimizer (SGD) + d_optimizer (SGD, lr×0.01) + scheduler + EMA + DDP
             └─ vòng epoch:
                  ├─ train_one_epoch(...)                        # ② LUỒNG HUẤN LUYỆN (chính)
                  │     for mỗi batch:
                  │        ├─ output,attn,proj_feat,proj_token = model(image)   # ③ forward HỌC SINH
                  │        ├─ tea_logits,tea_attn,tea_token,tea_feat = teacher(image) # ④ forward GIÁO VIÊN
                  │        ├─ D(real), D(fake)                    # ⑤ Discriminator
                  │        ├─ cls_loss / pca_loss / gl_loss / gan_loss          # ⑥ 4 LOSS
                  │        ├─ d_optimizer: backward(gan_loss); step()           # ⑦ cập nhật D
                  │        └─ optimizer: backward(loss); step()                 # ⑦ cập nhật HỌC SINH
                  ├─ lr_scheduler.step(); d_lr_scheduler.step()
                  ├─ evaluate(model, ...)                        # ⑧ ĐÁNH GIÁ
                  └─ save checkpoint
   (Suy luận: chỉ dùng nhánh CNN của model → ⑨)
```

Các luồng đối chiếu khác: `dist_train_student.py` (baseline) và `dist_train_logits.py` (logits KD) — Mục 12–13.

---

## 1. Luồng khởi động (entry point)

`experiments/run_cakd.sh`:

```bash
torchrun --nproc_per_node=8 dist_train_cakd.py --batch-size 32 --lr 0.1 \
  --lr-warmup-epochs 5 --lr-warmup-method linear \
  --auto-augment ta_wide --epochs 120 --random-erase 0.1 --mixup-alpha 0.2 \
  --weight-decay 0.00002 --norm-weight-decay 0.0 --label-smoothing 0.1 \
  --train-crop-size 224 --model-ema --val-resize-size 224 --ra-sampler --ra-reps 4 \
  --output-dir results/resnet50/cakd_vitb16
```

Diễn giải từng tham số quan trọng:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--nproc_per_node=8` | 8 | chạy **8 tiến trình = 8 GPU** (DDP). Batch toàn cục = 8×32 = **256** |
| `--batch-size 32` | 32 | ảnh **mỗi GPU** |
| `--lr 0.1` | 0,1 | learning rate khởi đầu (SGD) |
| `--lr-warmup-epochs 5 --lr-warmup-method linear` | | 5 epoch đầu **tăng dần lr** (tránh sốc) |
| `--auto-augment ta_wide` | TrivialAugmentWide | tăng cường mạnh ("đa khung nhìn" thay cho MVG) |
| `--random-erase 0.1` | 0,1 | xác suất xóa ngẫu nhiên một vùng (giống "patch mask" trong MVG) |
| `--mixup-alpha 0.2` | 0,2 | bật Mixup |
| `--epochs 120` | 120 | tổng số epoch (ImageNet) |
| `--label-smoothing 0.1` | 0,1 | làm mềm nhãn cho CrossEntropy |
| `--model-ema` | bật | giữ bản trọng số trung bình trượt |
| `--ra-sampler --ra-reps 4` | | Repeated Augmentation: mỗi ảnh xuất hiện 4 bản tăng cường khác nhau |

→ `torchrun` đặt biến môi trường `RANK/WORLD_SIZE/LOCAL_RANK`, gọi `dist_train_cakd.py` 8 lần (mỗi GPU một tiến trình), mỗi lần chạy `main(args)`.

> **Liên hệ bài báo:** `ta_wide + random-erase + mixup + ra-sampler` chính là phần hiện thực **"đa khung nhìn"** (công thức 7) bằng augmentor PyTorch chuẩn — vì bản công khai không kèm lớp `MVG` riêng.

---

## 2. Luồng dữ liệu (data pipeline)

Hàm `load_data(traindir, valdir, args)` và phần `DataLoader` trong `main`:

```
Thư mục ảnh (ImageFolder: mỗi lớp một thư mục con)
   │
   ▼  ClassificationPresetTrain (new_utils.py) — áp cho TỪNG ảnh:
   ├─ RandomResizedCrop(224)          # cắt-thu phóng ngẫu nhiên → (3,224,224)
   ├─ RandomHorizontalFlip(0.5)       # lật ngang 50%
   ├─ TrivialAugmentWide()            # auto-augment "ta_wide"
   ├─ PILToTensor + ConvertImageDtype # → float tensor
   ├─ Normalize(mean,std)             # chuẩn hóa theo ImageNet
   └─ RandomErasing(0.1)              # xóa vùng ngẫu nhiên
   │
   ▼  Sampler: RASampler (Repeated Augmentation, reps=4) + DistributedSampler
   │       → chia dữ liệu cho 8 GPU, mỗi ảnh lặp 4 bản augment khác nhau
   ▼  DataLoader(batch_size=32, collate_fn=...)
   │
   ▼  collate_fn = RandomChoice([RandomMixup(alpha=0.2), RandomCutmix])  (transforms.py)
       áp trên CẢ LÔ:
       - Mixup:  batch = λ·batch + (1−λ)·batch_rolled ; target trộn one-hot
       - CutMix: dán một vùng chữ nhật của ảnh khác vào ; target trộn theo diện tích
   │
   ▼  ĐẦU RA mỗi batch:
       image  : (B=32, 3, 224, 224)
       target : (B, num_classes)  ← one-hot/đã trộn (do Mixup/CutMix)
```

**Chi tiết Mixup** (`transforms.py`, `RandomMixup.forward`):

```python
if torch.rand(1) >= self.p: return batch, target      # 50% không trộn
batch_rolled = batch.roll(1, 0)                        # ghép ảnh i với ảnh i-1 (rẻ hơn shuffle)
lambda_param = Beta(alpha,alpha)                       # tỉ lệ trộn
batch = λ·batch + (1−λ)·batch_rolled                   # trộn ẢNH
target = λ·target + (1−λ)·target_rolled               # trộn NHÃN (one-hot)
```

**Chi tiết CutMix** (`RandomCutmix.forward`): chọn hộp `(x1,y1,x2,y2)` ngẫu nhiên, **dán vùng đó từ ảnh khác**, rồi `λ = 1 − diện_tích_hộp/diện_tích_ảnh` để trộn nhãn theo tỉ lệ diện tích.

> **Vì sao quan trọng cho CAKD:** dữ liệu mỗi batch đã rất "nhiễu/đa dạng" → tạo nhiều khung nhìn cho học sinh, làm nền cho phần huấn luyện đối kháng bền vững (Discriminator).

---

## 3. Luồng thiết lập mô hình (trong `main`)

```python
model         = torchvision.models.resnet50_cakd(num_classes=num_classes)      # HỌC SINH (sửa)
teacher       = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # GIÁO VIÊN (pretrain)
discriminator = new_utils.NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3)   # PatchGAN

criterion     = nn.CrossEntropyLoss(label_smoothing=0.1)   # cho cls_loss
mse_criterion = nn.MSELoss()                                # cho pca_loss, gl_loss
gan_criterion = new_utils.GANLoss()                         # BCEWithLogits cho GAN

optimizer   = SGD(model.params,        lr=0.1,      momentum=0.9, weight_decay=2e-5)
d_optimizer = SGD(discriminator.params, lr=0.01×0.1, momentum=0.9, ...)   # D HỌC CHẬM HƠN 100 lần
scaler      = GradScaler() if amp                            # mixed-precision (nếu --amp)
lr_scheduler/d_lr_scheduler = warmup(5 epoch) → StepLR/Cosine
model_ema   = ExponentialMovingAverage(model, decay≈0.9999)
# DDP: bọc model/teacher/discriminator bằng DistributedDataParallel (find_unused_parameters=True)
```

Các điểm cần nhớ về *luồng điều khiển*:

- **Giáo viên đóng băng:** luôn `teacher.eval()`, mọi tensor giáo viên đều `.detach()` khi vào loss ⇒ **không có gradient** chảy vào ViT.
- **Hai optimizer độc lập:** học sinh (`optimizer`) và D (`d_optimizer`) cập nhật riêng → đúng tinh thần GAN (Mục 7 file `04_GAN.md`).
- **`find_unused_parameters=True`:** cần vì khi suy luận/loss một số nhánh (vd `cls_proj`, projector) có thể không nhận gradient ở mọi bước.

---

## 4. Luồng huấn luyện 1 epoch — `train_one_epoch` (luồng chính CAKD)

Toàn bộ một bước (một batch) trong `dist_train_cakd.py`:

```python
model.train(); teacher.eval(); discriminator.train()
for image, target in data_loader:
    image, target = image.to(device), target.to(device)        # (B,3,224,224), (B,C)

    with autocast(enabled=scaler):
        # ── 4.1 forward HỌC SINH ─────────────────────────────
        output, attn_weights, proj_feat, proj_token = model(image)
        #   output       : (B,1000)        logits học sinh
        #   attn_weights : [attn_qk, attn_vv]  mỗi cái (B,196,196)
        #   proj_feat    : (B,196,768)     h'_S  (đầu ra GL)
        #   proj_token   : (B,768)         cls_proj(token CNN)

        # ── 4.2 forward GIÁO VIÊN ────────────────────────────
        tea_logits, tea_attn_weights, tea_token, tea_feat = teacher(image)
        #   tea_logits      : (B,1000)
        #   tea_attn_weights: 4 × (B,197,197)  [QK,VV kế-cuối ; QK,VV cuối]
        #   tea_token       : (B,768)
        #   tea_feat        : (B,196,768)  h_T

        # ── 4.3 chuẩn bị input cho Discriminator ─────────────
        input_d_real = tea_attn_weights[2][:,1:,1:].clone()[:,None,:,:].detach()  # (B,1,196,196) THẬT
        input_d_fake = attn_weights[0].clone()[:,None,:,:].detach()               # (B,1,196,196) GIẢ
        pred_real = discriminator(input_d_real)         # (B,1)
        pred_fake = discriminator(input_d_fake.detach())# (B,1)

        # ── 4.4 tính 4 LOSS ─────────────────────────────────
        cls_loss = criterion(output, target)            # CrossEntropy  (giám sát nhãn thật)
        pca_loss = 0.2 *MSE(attn_weights[0], tea_attn_weights[2][:,1:,1:].detach()) \
                 + 0.05*MSE(attn_weights[1], tea_attn_weights[3][:,1:,1:].detach())   # ── (4)
        gl_loss  = MSE(output, tea_logits.detach()) \
                 + MSE(proj_token, tea_token) \
                 + 0.05*MSE(proj_feat, tea_feat.detach())                             # ── (6)+thêm
        gan_loss = 0.5*(gan_criterion(pred_real.detach(),True)+gan_criterion(pred_fake,False)) # ── (8)
        λ = min(max(epoch-25,0)/50.0, 0.2)              # lịch warm-up cho phần chưng cất
        loss = cls_loss + λ*( pca_loss + gl_loss
                              + 0.05*gan_criterion(pred_real.detach(),True)
                              + gan_criterion(pred_fake,True) )                       # ── (10)+(9)

    # ── 4.5 BACKWARD LUÂN PHIÊN ─────────────────────────────
    d_optimizer.zero_grad()
    gan_loss.backward(retain_graph=True)   # ① cập nhật DISCRIMINATOR trước
    d_optimizer.step()

    optimizer.zero_grad()
    loss.backward()                        # ② cập nhật HỌC SINH (+PCA+GL) sau
    optimizer.step()

    # ── 4.6 EMA + đo đạc ────────────────────────────────────
    if model_ema and i % ema_steps == 0: model_ema.update_parameters(model)
    acc1, acc5 = accuracy(output, target); ... metric_logger.update(...)
```

**Vì sao thứ tự backward này?**
- `gan_loss.backward(retain_graph=True)`: cập nhật D bằng mục tiêu (8). `retain_graph=True` để giữ đồ thị tính toán cho lần backward thứ hai (`loss`).
- `loss.backward()`: cập nhật học sinh + projector bằng mục tiêu (10). Trong `loss`, các tensor giáo viên đều `.detach()`, và `pred_real.detach()` để phần generator **không** kéo D theo hướng sai.

**Vai trò `λ` (warm-up):**
```
epoch ≤ 25 : λ = 0     → CHỈ học cls_loss (CNN tự đứng vững)
26..75     : λ tăng tuyến tính 0 → 0.2
epoch ≥ 75 : λ = 0.2   → chưng cất ở cường độ tối đa
```

---

## 5. Luồng forward HỌC SINH — `ResNet_CAKD._forward_impl` (resnet.py)

Đi từng dòng, kèm shape (ảnh `(B,3,224,224)`):

```python
x = conv1(x); x = bn1(x); x = relu(x); x = maxpool(x)   # → (B,  64, 56, 56)
x   = self.layer1(x)                                     # → (B, 256, 56, 56)
x   = self.layer2(x)                                     # → (B, 512, 28, 28)
x_3 = self.layer3(x)                                     # → (B,1024, 14, 14)  ← h_S

# —— "token hóa" đặc trưng CNN ——
tmp = torch.reshape(x_3, (B,1024,-1))                    # → (B,1024,196)
tmp = tmp.permute(0,2,1)                                 # → (B,196,1024)  (giống chuỗi token ViT)

# —— NHÁNH PCA ——
_, attn_qk, attn_vv = self.pca_proj(tmp)                 # mỗi cái (B,16,196,196)  (16 đầu)
num_heads = attn_qk.shape[1]                             # 16
attn_qk = attn_qk.sum(1)/num_heads                       # → (B,196,196)  gộp đầu
attn_vv = attn_vv.sum(1)/num_heads                       # → (B,196,196)

# —— NHÁNH GL ——
vit_feat = self.gl_proj(tmp)                             # → (B,196,768)  h'_S

# —— NHÁNH PHÂN LOẠI (vẫn dùng x_3) ——
x = self.layer4(x_3)                                     # → (B,2048, 7, 7)
x = self.avgpool(x)                                      # → (B,2048, 1, 1)
cnn_token = torch.flatten(x,1)                           # → (B,2048)
x = self.fc(cnn_token)                                   # → (B,1000)  logits

return x, [attn_qk, attn_vv], vit_feat, self.cls_proj(cnn_token)
#      └logits┘ └─ cho (4) ─┘  └(6)┘     └─ (B,768) cho gl_loss(token) ─┘
```

**Điểm tinh ý của luồng:**
- Projector "rẽ nhánh" sau `layer3`, nhưng `layer4 → fc` vẫn chạy trên `x_3` ⇒ **một forward, hai nhiệm vụ** (phân loại + chưng cất).
- Lưới `14×14 = 196` được *cố ý* ép thành 196 token để khớp số patch ViT.

---

## 6. Luồng forward GIÁO VIÊN — ViT (`vision_transformer.py` + `functional.py`)

```python
# VisionTransformer.forward(x):  x = (B,3,224,224)
x = self._process_input(x)             # chia patch + Linear → (B,196,768)
batch_class_token = class_token.expand(B,-1,-1)        # (B,1,768)
x = torch.cat([batch_class_token, x], dim=1)           # (B,197,768)  thêm class token
x, attn_weights = self.encoder(x)      # qua 12 block; lấy attn 2 block cuối
cls_token = x[:, 0]                     # (B,768)  token lớp
feats     = x[:, 1:]                    # (B,196,768)  h_T
x = self.heads(cls_token)              # (B,1000)  MLP Head
return x, attn_weights, cls_token, feats
```

**Bên trong Encoder** (`Encoder.forward`):

```python
for i in range(num_layers):            # 12 block
    if i < num_layers-2:   x = layers[i](x)                 # block thường (không trả attn)
    elif i == num_layers-2: x, attn_weights_2 = layers[i](x, True)   # block kế cuối
    else:                   x, attn_weights_1 = layers[i](x, True)   # block cuối
return ln(x), [attn_weights_2[0], attn_weights_2[1],       # [0],[1] QK,VV kế-cuối
               attn_weights_1[0], attn_weights_1[1]]        # [2],[3] QK,VV cuối
```

**Bên trong attention** (`functional.py`, đã sửa) — chỗ tạo ra `QK^T` và `VV^T`:

```python
q = q / sqrt(E)                        # chia tỉ lệ trước
attn    = bmm(q, k.transpose(-2,-1))   # = QK^T/√d   (B·h,197,197)  ← TRƯỚC softmax
attn_vv = bmm(v, v.transpose(-2,-1))   # = VV^T
attn_soft = softmax(attn,-1)           # softmax CHỈ để tính đầu ra
output  = bmm(attn_soft, v)
return output, [attn, attn_vv]         # trả về điểm số THÔ cho KD
```

> **Khớp với học sinh:** học sinh cũng trả `dots_qk = QK^T·scale` và `dots_vv = VV^T·scale` (trước softmax). Vậy **cả hai phía đều so khớp ma trận điểm số trước softmax** — đó là "bản đồ chú ý" trong công thức (4).
>
> `tea_attn_weights[2]`/`[3]` (block cuối) có kích thước `197×197` (kèm class token); khi vào loss bị cắt `[:,1:,1:]` → `196×196` để khớp học sinh.

---

## 7. Luồng PCA projector — `Attention.forward` (resnet.py)

Đầu vào `tmp = (B,196,1024)`. `dim=1024, heads=16, dim_head=64`.

```python
qkv = self.to_qkv(x).chunk(3, dim=-1)           # 1 Linear: 1024 → 3072, rồi tách 3 → mỗi cái (B,196,1024)
q,k,v = map(lambda t: rearrange(t,'b n (h d)->b h n d',h=16), qkv)   # → (B,16,196,64)

dots_qk = matmul(q, k.transpose(-1,-2)) * scale # (B,16,196,196)  = Q K^T/√d   ── công thức (2)
dots_vv = matmul(v, v.transpose(-1,-2)) * scale # (B,16,196,196)  = V V^T/√d   ── cho (4) số hạng 2
attn_qk = softmax(dots_qk)                       # (B,16,196,196)
out = matmul(attn_qk, v)                          # (B,16,196,64)  đầu ra attention (BỊ BỎ qua "_")
out = rearrange(out,'b h n d->b n (h d)')         # (B,196,1024)
return self.to_out(out), dots_qk, dots_vv         # chỉ dots_qk, dots_vv được dùng tiếp
```

**Luồng dữ liệu PCA tóm tắt:**
```
h_S (B,196,1024) ──to_qkv──► Q,K,V (B,16,196,64)
                              ├─ QK^T/√d → dots_qk (B,16,196,196) ─(gộp đầu)→ attn_qk (B,196,196)
                              └─ VV^T/√d → dots_vv (B,16,196,196) ─(gộp đầu)→ attn_vv (B,196,196)
```

> **Ánh xạ bài báo:** đây là công thức (1)–(2). Lưu ý: mẹo **Partially-Cross (công thức 3)** — thay ngẫu nhiên Q/K/V học sinh bằng giáo viên — **KHÔNG có** trong lớp `Attention` công khai; code chỉ tính attention thuần rồi MSE với giáo viên.

---

## 8. Luồng GL projector — `GLProj.forward` (resnet.py)

Đầu vào `tmp = (B,196,1024)`. `num_patch=196 → num_fc=16`. Mỗi lớp FC: `1024 → 768`.

```python
out = torch.zeros((B, 196, 768)).cuda()          # khung kết quả rỗng
for i in range(16):                              # 16 nhóm patch
    out[:, idx[i], :] = self.layers[i](x[:, idx[i], :])   # FC_i chỉ xử lý các patch thuộc nhóm i
return out                                        # (B,196,768)  = h'_S
```

**`idx[i]` là gì?** Danh sách chỉ số patch của **một lân cận 4×4** trên lưới 14×14. Ví dụ nhóm 0:

```python
idx_224_16_0 = [0,1,2,3, 14,15,16,17, 28,29,30,31, 42,43,44,45]
#               hàng 0     hàng 1        hàng 2        hàng 3   → một khối 4×4 góc trên-trái
```

```
Lưới 14×14 (196 patch) chia thành 16 nhóm; mỗi nhóm ~ một mảng vuông dùng CHUNG 1 lớp FC:
┌────┬────┬────┬──┐
│ G0 │ G1 │ G2 │G3│   →  thay vì 196 FC riêng (rất tốn) → chỉ 16 FC (gọn nhẹ)
├────┼────┼────┼──┤
│ G4 │ G5 │ G6 │G7│
├────┼────┼────┼──┤
│ G8 │ G9 │G10 │..│
└────┴────┴────┴──┘
```

> **Ánh xạ bài báo:** công thức (5) `h'_S = Proj2(h_S)` + ý "một lân cận 4×4 dùng chung một FC" → còn 16 FC. Khi vào loss: `0.05·MSE(proj_feat, tea_feat)` = công thức (6).

---

## 9. Luồng Discriminator + GANLoss + backward luân phiên

### 9.1 Discriminator — `NLayerDiscriminator.forward` (new_utils.py)

Đầu vào: bản đồ chú ý **1 kênh** `(B,1,196,196)`.

```
(B,1,196,196)
 ─ Conv 4×4 s2 (1→8)   + LeakyReLU(0.2)   → (B,  8, 98, 98)
 ─ Conv 4×4 s2 (8→16)  + BN + LeakyReLU   → (B, 16, 49, 49)
 ─ Conv 4×4 s2 (16→32) + BN + LeakyReLU   → (B, 32, 24, 24)
 ─ Conv 4×4 s1 (32→64) + BN + LeakyReLU   → (B, 64, 23, 23)
 ─ Conv 4×4 s1 (64→1)                     → (B,  1, 22, 22)  bản đồ điểm
 ─ AdaptiveAvgPool2d(1)                   → (B,  1,  1,  1) → (B,1)  điểm thật/giả
```

### 9.2 GANLoss — `GANLoss.__call__` (new_utils.py)

```python
# gan_mode='vanilla' → BCEWithLogitsLoss
target = 1.0 nếu target_is_real else 0.0           # tự tạo nhãn cùng shape với pred
loss = BCEWithLogitsLoss(prediction, target)
```

### 9.3 Hai mục tiêu đối kháng

```
D (qua d_optimizer):  gan_loss = 0.5*( BCE(D(thật),1) + BCE(D(giả),0) )           ── (8) L_MAD
Học sinh (generator): ...+ 0.05*BCE(D(thật).detach(),1) + BCE(D(giả),1)           ── (9) L_MVG (non-saturating)
```

### 9.4 Trình tự backward (rất quan trọng)

```
① d_optimizer.zero_grad()
   gan_loss.backward(retain_graph=True)   # chỉ cập nhật tham số D
   d_optimizer.step()

② optimizer.zero_grad()
   loss.backward()                        # cập nhật Θ_S + PCA + GL (không đụng D do detach)
   optimizer.step()
```

- `input_d_fake.detach()` khi tính `pred_fake` cho D ⇒ gradient của `gan_loss` **không** chảy về học sinh (D chỉ học phân biệt).
- Trong `loss` của học sinh, `pred_real.detach()` ⇒ phần thật không kéo gì; còn `gan_criterion(pred_fake,True)` **cho phép** gradient chảy ngược qua D về **học sinh** (qua `input_d_fake = attn_weights[0]`), đẩy học sinh tạo attention "giống thật".

> ⚠️ **Khác biệt paper↔code:** paper cho D phân biệt **đặc trưng** `h_T`/`h'_S`; code cho D phân biệt **bản đồ chú ý** `(B,1,196,196)`.

### 9.5 Tổng hợp 4 loss ↔ công thức

| Loss (code) | Thành phần | Công thức | Trọng số |
|---|---|---|---|
| `cls_loss` | CE(output, target) | (ngoài 10) | 1 |
| `pca_loss` | MSE(attn_qk, QK_T) + MSE(attn_vv, VV_T) | **(4)** | 0.2 / 0.05 |
| `gl_loss` | MSE(logits)+MSE(token)+MSE(feat) | **(6)**+thêm | 1 /1 /0.05 |
| `gan_loss` | BCE(real,1)+BCE(fake,0) | **(8)** | 0.5 |
| `loss` | cls + λ·(pca+gl+gan_gen) | **(10)**+(9) | λ∈[0,0.2] |

---

## 10. Luồng đánh giá — `evaluate`

```python
model.eval()
with torch.inference_mode():                       # tắt gradient
    for image, target in data_loader_test:
        output, _, _, _ = model(image)             # CHỈ lấy logits, BỎ attn/feat/token
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1,5))
    ...reduce_across_processes(...)                 # gộp số liệu từ 8 GPU
print(f"Acc@1 {...} Acc@5 {...}")
```

- Luồng đánh giá **bỏ qua hoàn toàn** projector/discriminator — chỉ dùng nhánh phân loại CNN.
- `inference_mode()` nhanh hơn `no_grad()` (không lưu đồ thị).
- Có cả `evaluate(model_ema, ...)` để đo bản EMA (thường nhỉnh hơn).

---

## 11. Luồng suy luận (inference / triển khai)

```
Sau hội tụ → bỏ Proj1(PCA), Proj2(GL), Discriminator, ViT giáo viên.
Triển khai CHỈ nhánh CNN:
   image (B,3,224,224)
     → conv1..layer4 → avgpool → fc → logits (B,1000) → argmax → lớp
```

- Vì projector nằm "rẽ nhánh" sau `layer3` và **không** nằm trên đường `layer4→fc`, nên khi bỏ chúng, **đường phân loại không đổi** ⇒ **0 chi phí thêm** lúc suy luận. Đây là lý do CAKD hấp dẫn cho thiết bị biên (CUDA/TensorRT/NCNN).

---

## 12. Luồng baseline — `dist_train_student.py` (đối chiếu)

```python
model = torchvision.models.resnet50(num_classes=num_classes)   # ResNet50 GỐC (trả 1 thứ)
...
output = model(image)                 # (B,1000)
loss   = criterion(output, target)    # chỉ CrossEntropy
loss.backward(); optimizer.step()
```

- **Không** teacher, **không** projector, **không** GAN. Đây là mốc so sánh "học chay".
- Kết quả (README): ResNet50 baseline = **73,82%** Top-1.

---

## 13. Luồng logits KD — `dist_train_logits.py` (đối chiếu)

```python
model   = resnet50_cakd(...)          # vẫn trả 4 thứ (nhưng dùng ít)
teacher = vit_b_16(pretrained); teacher.eval()
...
output, attn_weights, proj_feat, proj_token = model(image)
tea_logits, tea_attn_weights, tea_token, tea_feat = teacher(image)
logits_loss = mse_criterion(output, tea_logits.detach())   # CHỈ khớp logits
cls_loss    = criterion(output, target)
loss = cls_loss + logits_loss          # KHÔNG PCA, KHÔNG GL, KHÔNG GAN
```

- Đây là KD "thường" (chỉ logits, dạng MSE). Kết quả (README): **74,48%** Top-1.
- **So sánh ba luồng (ImageNet, ResNet50):**

| Luồng | Thành phần KD | Top-1 |
|---|---|---|
| baseline (`student`) | — | 73,82% |
| logits (`logits`) | MSE(logits) | 74,48% |
| **CAKD (`cakd`)** | PCA + GL + GAN | **76,21%** |

→ Phần "đắt giá" (PCA+GL+GAN) đem lại **+1,73%** so với logits, **+2,39%** so với baseline.

---

## 14. Luồng thiết lập môi trường — chép đè torch (README, bước 2)

Vì `nn.MultiheadAttention` gốc **không trả về** `QK^T`/`VV^T`, project phải **chép đè 3 file** của torch trước khi chạy:

```shell
cp cakd_modified_files/resnet.py              ${TORCHVISION_MODEL_PATH}/resnet.py
cp cakd_modified_files/vision_transformer.py  ${TORCHVISION_MODEL_PATH}/vision_transformer.py
cp cakd_modified_files/functional.py          ${TORCH_NN_PATH}/functional.py
```

Sau khi chép đè:
- `torchvision.models.resnet50_cakd` **mới tồn tại** (đăng ký trong `resnet.py` đã sửa).
- `vit_b_16` **trả về 4 thứ** (logits, attn, token, feat) thay vì chỉ logits.
- `F.multi_head_attention_forward` **trả thêm** `[QK, VV]` trước softmax.

> Đây là lý do code không chạy được nếu bỏ qua bước chép đè — các API `resnet50_cakd` và chữ ký `forward` mở rộng đến từ chính các file này.

---

## 15. Bảng tổng: luồng ↔ thành phần ↔ công thức

| # | Luồng | File / hàm | Đầu vào → Đầu ra (shape) | Công thức |
|---|---|---|---|---|
| ① | Khởi động | `run_cakd.sh` → `main` | tham số → tiến trình 8 GPU | — |
| ② | Dữ liệu | `load_data`, `transforms.py` | thư mục ảnh → `(B,3,224,224)`,`(B,C)` | (7) (qua augment) |
| ③ | Thiết lập | `main` | — → model/teacher/D/optim | — |
| ④ | Huấn luyện | `train_one_epoch` | batch → 4 loss → cập nhật | (10) |
| ⑤ | Forward học sinh | `ResNet_CAKD._forward_impl` | `(B,3,224,224)` → `(logits, [attn], h'_S, token)` | (1)(2)(5) |
| ⑥ | Forward giáo viên | `VisionTransformer.forward` | `(B,3,224,224)` → `(logits, 4×attn, token, h_T)` | — |
| ⑦ | PCA | `Attention.forward` | `(B,196,1024)` → 2×`(B,196,196)` | (1)–(4) |
| ⑧ | GL | `GLProj.forward` | `(B,196,1024)` → `(B,196,768)` | (5)(6) |
| ⑨ | Discriminator | `NLayerDiscriminator` | `(B,1,196,196)` → `(B,1)` | (8)(9) |
| ⑩ | Loss + backward | `train_one_epoch` | tensors → loss; D rồi student | (4)(6)(8)(9)(10) |
| ⑪ | Đánh giá | `evaluate` | test batch → Acc@1/5 | — |
| ⑫ | Suy luận | nhánh CNN | `(B,3,224,224)` → `(B,1000)` | — |
| ⑬ | Baseline | `dist_train_student.py` | ảnh → CE | — |
| ⑭ | Logits KD | `dist_train_logits.py` | ảnh → CE + MSE(logits) | KD logits |

---

## 16. Sơ đồ luồng dữ liệu một batch (toàn cảnh)

```
                                   image (B,3,224,224)
                    ┌───────────────────┴───────────────────┐
            (đóng băng) ViT                            ResNet50_CAKD
      patch→+cls+pos→12 block                 conv→layer1→layer2→layer3=x_3 (B,1024,14,14)
        │      │     │     │                          │ reshape+permute → (B,196,1024)
   logits  4×attn  token  h_T                  ┌──────┼───────────┐
   (B,1000)(197²)(B,768)(196,768)          PCA│   GL │       layer4→avgpool→fc
        │      │     │     │            (196²)│(196,768)        │        │
        │      │     │     │           attn_qk│ vit_feat   logits(B,1000) cls_proj(B,768)
        ▼      ▼     ▼     ▼               ▼   ▼            ▼              ▼
      MSE(gl logits) MSE(token)  MSE(feat) MSE(qk) MSE(vv)  CrossEntropy(nhãn)
            └───────── gl_loss ──────────┘ └── pca_loss ──┘     cls_loss
      attn_qk(giả) ─► D ◄─ attn_T(thật) ─► gan_loss / generator-loss
                          Σ → loss (10) → backward → cập nhật học sinh
```

---

*Hết. Xem `Phan_tich_chi_tiet_CAKD.md` cho giải thích công thức, và `TaiLieu_NenTang/` cho kiến thức nền của từng khối.*



