# Chặng 4 — `CAKD/dist_train_cakd.py` (GHÉP TẤT CẢ)

> File train chính. Nếu đã đọc chặng 1–3 thì mọi biến ở đây đều có nghĩa.
> Đọc kỹ `train_one_epoch` (18–85) — đây là đỉnh của cả dự án. `main` đọc theo khối. `get_args_parser` ⚪ lướt.

---

## 🟢 `train_one_epoch` — dòng 18–85  (TRÁI TIM)

### Chuẩn bị (18–30)
```python
def train_one_epoch(model, discriminator, teacher, mse_criterion, gan_criterion, criterion,
                    optimizer, d_optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()          # HỌC SINH: bật chế độ train (dropout/BN cập nhật)
    teacher.eval()         # GIÁO VIÊN: đóng băng (không cập nhật thống kê BN, không dropout)
    discriminator.train()  # DISCRIMINATOR: bật train
    metric_logger = new_utils.MetricLogger(delimiter="  ")   # bộ ghi log
    metric_logger.add_meter("lr", ...)                       # khai báo các chỉ số sẽ in: lr, img/s, các loss
    ...
    header = f"Epoch: [{epoch}]"
```

### Vòng lặp từng batch (31–49) — forward & lắp loss
```python
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)     # đưa dữ liệu lên GPU
        with torch.cuda.amp.autocast(enabled=scaler is not None):   # bật mixed-precision nếu có scaler
            # ── FORWARD 2 nhánh ──
            output, attn_weights, proj_feat, proj_token = model(image)          # HỌC SINH → 4 thứ
            tea_logits, tea_attn_weights, tea_token, tea_feat = teacher(image)  # GIÁO VIÊN → 4 thứ

            # ── Chuẩn bị đầu vào cho Discriminator (attention map coi như ảnh 1 kênh) ──
            input_d_real = tea_attn_weights[2][:, 1:, 1:].clone()[:, None, :, :].detach()  # attn qk lớp cuối của THẦY (bỏ CLS) → "thật"
            input_d_fake = attn_weights[0].clone()[:, None, :, :].detach()                 # attn qk của TRÒ           → "giả"
            #   [:,1:,1:] bỏ hàng/cột CLS (197→196) ; [:,None,:,:] thêm chiều kênh =1 ; .detach() chặn gradient rò sang model/teacher

            pred_real = discriminator(input_d_real)               # D chấm điểm map "thật"
            pred_fake = discriminator(input_d_fake.detach())      # D chấm điểm map "giả"

            # ── 4 THÀNH PHẦN LOSS ──
            cls_loss = criterion(output, target)                  # (1) CrossEntropy phân loại — mạch giám sát nhãn thật

            pca_loss = 0.2  * mse_criterion(attn_weights[0], tea_attn_weights[2][:,1:,1:].detach())  \
                     + 0.05 * mse_criterion(attn_weights[1], tea_attn_weights[3][:,1:,1:].detach())
            #   khớp attention TRÒ↔THẦY: qk (trọng số 0.2) + vv (trọng số 0.05)   → PCA

            gl_loss = mse_criterion(output, tea_logits.detach()) \
                    + mse_criterion(proj_token, tea_token) \
                    + 0.05 * mse_criterion(proj_feat, tea_feat.detach())
            #   khớp logits + token CLS + đặc trưng patch                         → GL

            gan_loss = 0.5 * (gan_criterion(pred_real.detach(), True) + gan_criterion(pred_fake, False))
            #   loss dạy DISCRIMINATOR: D(thật)→1, D(giả)→0                       → nuôi backward (1)

            # ── TỔNG LOSS của HỌC SINH ──
            loss = cls_loss + min(max(epoch-25, 0)/50.0, 0.2) * 1.0 * (
                       pca_loss + gl_loss
                     + 0.05 * gan_criterion(pred_real.detach(), True)   # ← .detach() ⇒ gradient=0 (no-op, chỉ ghi sổ)
                     + gan_criterion(pred_fake, True) )                 # ← TRÒ "lừa" D coi map mình là thật (non-saturating)
            #          └── λ: 0 trong 25 epoch đầu, tăng tới 0.2 ở epoch 35 rồi giữ ──
```
> **Đọc dòng `loss` này như sau:**
> - `cls_loss` luôn có (ngoài λ) → 25 epoch đầu học sinh CHỈ học phân loại.
> - `λ · (pca + gl + gan)` → phần chưng cất, bị "công tắc" λ bật dần.
> - Số hạng `0.05*gan(pred_real.detach(),...)` **vô tác dụng** với gradient học sinh (đã detach) — bỏ đi kết quả không đổi.

### Hai lần backward RIÊNG (51–68)
```python
        # ── BACKWARD (1): cập nhật DISCRIMINATOR ──  (KHÔNG bị λ chặn → chạy mọi batch từ epoch 0)
        d_optimizer.zero_grad()
        gan_loss.backward(retain_graph=True)   # retain_graph vì graph còn dùng cho backward (2)
        d_optimizer.step()

        # ── BACKWARD (2): cập nhật HỌC SINH ──
        optimizer.zero_grad()
        if scaler is not None:                 # nhánh AMP (mixed precision)
            scaler.scale(loss).backward()      # nhân tỉ lệ để tránh underflow rồi backward
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)     # bỏ tỉ lệ trước khi clip
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)             # cập nhật trọng số
            scaler.update()                    # điều chỉnh hệ số scale cho bước sau
        else:                                  # nhánh thường (không AMP)
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
```
> Vì sao update D **trước**? Để D "đứng vững" nhận diện thật/giả trước, rồi học sinh mới cố lừa nó ở backward (2).
> `retain_graph=True` là bắt buộc: cùng một forward graph được backward 2 lần (gan_loss rồi loss).

### EMA + ghi log (70–85)
```python
        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)          # cập nhật bản trung bình trượt của model
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)           # trong warmup: copy thẳng weight (reset bộ đếm)

        acc1, acc5 = new_utils.accuracy(output, target, topk=(1,5))   # đo độ chính xác
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])  # ghi log các chỉ số
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        ... (acc5, pca_loss, cls_loss, gl_loss, img/s, gan_loss) ...
```

---

## 🟢 `evaluate` — dòng 88–128
```python
def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()                                    # tắt dropout/BN-update
    with torch.inference_mode():                    # không lưu graph → nhanh, đỡ tốn RAM
        for image, target in ...:
            output, _, _, _ = model(image)          # ← CHỈ lấy logits, bỏ 3 output KD
            loss = criterion(output, target)
            acc1, acc5 = new_utils.accuracy(output, target, topk=(1,5))
            ...
    num_processed_samples = new_utils.reduce_across_processes(...)   # gộp số liệu các GPU
    metric_logger.synchronize_between_processes()                    # đồng bộ log giữa các process
    print(f"... Acc@1 ... Acc@5 ...")
    return metric_logger.acc1.global_avg
```
> Điểm cần nhớ: lúc **đánh giá/suy luận, chỉ dùng logits** → mô hình triển khai thực tế nhẹ như ResNet thường,
> mọi thứ PCA/GL/GAN chỉ tồn tại lúc train.

---

## 🟢 `main` — dòng 219–437 (đọc theo KHỐI, không cần từng dòng)

| Dòng | Khối | Làm gì |
|------|------|--------|
| 220–235 | Khởi tạo | tạo thư mục out, bật distributed, chọn device, đường dẫn data |
| 236–261 | Dữ liệu | `load_data` → dataset/sampler; dựng `DataLoader`; Mixup/CutMix qua `collate_fn` |
| 263–269 | **3 mô hình** | `resnet50_cakd` (trò), `vit_b_16` pretrained (thầy), `NLayerDiscriminator(input_nc=1,ndf=8)` |
| 275–277 | 3 loss | `criterion`=CE, `mse_criterion`=MSE, `gan_criterion`=GANLoss |
| 279–310 | **2 optimizer** | `optimizer` (trò), `d_optimizer` (D, lr = 0.01×lr) |
| 311 | AMP | tạo `scaler` nếu `--amp` |
| 313–360 | Scheduler | mỗi optimizer 1 scheduler + warmup riêng |
| 362–371 | DDP | bọc model cho đa GPU (`find_unused_parameters=True` vì nhánh PCA/GL) |
| 373–384 | EMA | tạo `ExponentialMovingAverage` nếu `--model-ema` |
| 386–406 | Resume/test | nạp checkpoint / chạy `--test-only` |
| 408–433 | **VÒNG EPOCH** | mỗi epoch: `train_one_epoch → 2×scheduler.step → evaluate → save checkpoint` |

```python
    for epoch in range(args.start_epoch, args.epochs):        # dòng 410
        if args.distributed: train_sampler.set_epoch(epoch)
        train_one_epoch(model, discriminator, teacher, mse_criterion, gan_criterion, criterion,
                        optimizer, d_optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step(); d_lr_scheduler.step()            # giảm LR cho cả 2
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema: evaluate(model_ema, ...)                # đánh giá thêm bản EMA
        if args.output_dir:                                  # lưu checkpoint (mỗi 10 epoch lưu bản riêng)
            ...
```

---

## ⚪ `get_args_parser` — dòng 440–567 (LƯỚT)
Chỉ là khai báo tham số dòng lệnh. Khi cần biết `--lr`, `--epochs`, `--model-ema`… thì tra ở đây.
Giá trị thực tế khi chạy nằm trong `CAKD/experiments/run_cakd.sh`.

---

## ✅ Kiểm tra hiểu (nếu trả lời được hết là đã nắm toàn bộ CAKD)
1. Trong một batch, `backward` được gọi mấy lần, mỗi lần cập nhật mô hình nào?
2. Vì sao cần `retain_graph=True` ở backward (1)?
3. Ở epoch 10, giá trị λ là bao nhiêu → học sinh thực chất đang học loss nào?
4. Vì sao mọi output của teacher đều `.detach()` trước khi vào loss?
5. Lúc `evaluate`, vì sao chỉ lấy `output` mà bỏ 3 thứ còn lại?
