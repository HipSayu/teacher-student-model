# =============================================================================
# dist_train_cakd.py — SCRIPT HUẤN LUYỆN CHÍNH của CAKD ("nhạc trưởng")
# -----------------------------------------------------------------------------
# File này RÁP mọi thứ lại và chạy training:
#   - student   = ResNet_CAKD  (resnet50_cakd)  <- con model ta muốn dạy cho giỏi
#   - teacher   = ViT-B/16 pretrain             <- thầy giáo, ĐÓNG BĂNG (eval)
#   - discriminator = NLayerDiscriminator (GAN) <- "giám khảo" phân biệt attention thật/giả
#
# Ý tưởng: student vừa học phân loại (nhãn thật) VỪA bắt chước teacher qua 4 loss:z
#   cls_loss  : phân loại đúng nhãn (CrossEntropy)
#   pca_loss  : khớp attention map student <-> teacher (MSE)
#   gl_loss   : khớp logits + token + feature student <-> teacher (MSE)
#   gan_loss  : ép attention student "trông như thật" (đối kháng GAN)
#
# Có 2 optimizer chạy xen kẽ (kiểu GAN):
#   d_optimizer -> dạy discriminator phân biệt thật/giả
#   optimizer   -> dạy student (đánh lừa discriminator + khớp teacher + phân loại)
#
# CẤU TRÚC FILE:
#   train_one_epoch(): 1 epoch huấn luyện (TRÁI TIM — chứa toàn bộ logic loss & GAN)
#   evaluate():        đánh giá accuracy trên tập test
#   load_data():       nạp & tiền xử lý dữ liệu ImageNet
#   main():            khởi tạo model/optimizer/scheduler + vòng lặp epoch + lưu checkpoint
#   get_args_parser(): khai báo tham số dòng lệnh (argparse)
# =============================================================================

import datetime
import os
import time
import warnings

import new_utils          # hộp đồ nghề: GANLoss, Discriminator, MetricLogger, accuracy... (file đã đọc)
import torch
import torch.utils.data
import torchvision
import transforms          # các phép augment mixup/cutmix riêng của project
from new_utils import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import ViT_B_16_Weights   # bộ trọng số pretrain của teacher ViT-B/16


# =============================================================================
# train_one_epoch — HUẤN LUYỆN 1 EPOCH (nơi tính loss & cập nhật trọng số)
# =============================================================================
def train_one_epoch(model, discriminator, teacher, mse_criterion, gan_criterion, criterion, optimizer, d_optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()          # student: bật chế độ train (bật dropout, cập nhật BatchNorm)
    teacher.eval()         # teacher: chế độ eval -> ĐÓNG BĂNG, không học, chỉ phát tín hiệu
    discriminator.train()  # discriminator: bật chế độ train
    # Thiết lập bộ ghi log để theo dõi các loss thành phần
    metric_logger = new_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", new_utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", new_utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("pca_loss", new_utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("gl_loss", new_utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("cls_loss", new_utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("gan_loss", new_utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    # Duyệt từng batch ảnh; log_every vừa lặp vừa in tiến độ + ETA
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)   # đưa dữ liệu lên GPU
        # autocast: tự động dùng float16 ở chỗ an toàn -> nhanh hơn, đỡ tốn RAM (mixed precision)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # ---- CHẠY XUÔI CẢ HAI MODEL ----
            # Student trả 4 thứ: logits, [attn_qk, attn_vv], vit_feat, cls_token (xem resnet.py)
            output, attn_weights, proj_feat, proj_token = model(image)
            # Teacher trả 4 thứ: logits, 4 attention map, cls_token, feats (xem vision_transformer.py)
            tea_logits, tea_attn_weights, tea_token, tea_feat = teacher(image)

            # ---- CHUẨN BỊ ĐẦU VÀO CHO DISCRIMINATOR (GAN) ----
            # "Thật" = attention của teacher (bỏ class token bằng [:, 1:, 1:] -> còn 196x196),
            #          [:, None, :, :] chèn chiều kênh =1 -> shape (batch,1,196,196) cho Conv2d.
            #          detach() = cắt gradient (đây là "mẫu thật" cố định).
            input_d_real = tea_attn_weights[2][:, 1:, 1:].clone()[:, None, :, :].detach()
            # "Giả" = attention của student (attn_qk). detach() vì lúc dạy discriminator không sửa student.
            input_d_fake = attn_weights[0].clone()[:, None, :, :].detach()

            # Discriminator chấm điểm thật/giả
            pred_real = discriminator(input_d_real)              # điểm cho attention teacher
            pred_fake = discriminator(input_d_fake.detach())     # điểm cho attention student

            # ---- TÍNH 4 LOSS THÀNH PHẦN ----
            # 1) cls_loss: phân loại đúng nhãn thật (CrossEntropy) — nhiệm vụ chính của student
            cls_loss = criterion(output, target)
            # 2) pca_loss: ép attention student GIỐNG teacher (MSE). Dùng 2 attention (qk & vv) với
            #    trọng số 0.2 và 0.05. .detach() ở phía teacher vì teacher không học.
            pca_loss = 0.2 * mse_criterion(attn_weights[0], tea_attn_weights[2][:, 1:, 1:].detach()) + 0.05 * mse_criterion(attn_weights[1], tea_attn_weights[3][:, 1:, 1:].detach())
            # 3) gl_loss: ép logits/token/feature student giống teacher (KD "mềm" + khớp đặc trưng)
            gl_loss = mse_criterion(output, tea_logits.detach()) +  mse_criterion(proj_token, tea_token) + 0.05 * mse_criterion(proj_feat, tea_feat.detach())
            # 4) gan_loss (dạy DISCRIMINATOR): teacher -> True (thật), student -> False (giả).
            #    Đây là loss để CẬP NHẬT DISCRIMINATOR, không phải student.
            gan_loss = 0.5 * (gan_criterion(pred_real.detach(), True) + gan_criterion(pred_fake, False))

            # ---- TỔNG LOSS CHO STUDENT ----
            # Hệ số min(max(epoch-25,0)/50, 0.2): "khởi động chậm" — 25 epoch đầu chỉ học phân loại
            # (hệ số=0), rồi TĂNG DẦN ảnh hưởng của distill, chặn trần ở 0.2. Tránh làm hỏng student sớm.
            # Trong ngoặc: pca_loss + gl_loss + phần GAN ép student đánh lừa discriminator
            # (gan_criterion(pred_fake, True) = student MUỐN discriminator tưởng attention của nó là THẬT).
            loss = cls_loss + min(max(epoch-25, 0)/50.0, 0.2) * 1.0 * (pca_loss + gl_loss + 0.05 * gan_criterion(pred_real.detach(), True) + gan_criterion(pred_fake, True))

        # ---- BƯỚC 1: CẬP NHẬT DISCRIMINATOR ----
        d_optimizer.zero_grad()                  # xóa gradient cũ
        gan_loss.backward(retain_graph=True)     # lan truyền ngược gan_loss; giữ lại đồ thị để backward loss kế
        d_optimizer.step()                       # cập nhật trọng số discriminator

        # ---- BƯỚC 2: CẬP NHẬT STUDENT ----
        optimizer.zero_grad()
        if scaler is not None:
            # Nhánh mixed-precision (AMP): scale loss để tránh underflow float16
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # Nếu cắt gradient thì phải "unscale" trước
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Nhánh thường (float32)
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)   # cắt gradient chống bùng nổ
            optimizer.step()

        # ---- CẬP NHẬT MODEL EMA (bản trung bình mượt của student, nếu bật) ----
        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Trong giai đoạn warmup: reset để EMA cứ copy thẳng trọng số (chưa làm mượt)
                model_ema.n_averaged.fill_(0)

        # ---- GHI LOG SỐ LIỆU ----
        acc1, acc5 = new_utils.accuracy(output, target, topk=(1, 5))   # accuracy top-1/top-5
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["pca_loss"].update(pca_loss.item(), n=batch_size)
        metric_logger.meters["cls_loss"].update(cls_loss.item(), n=batch_size)
        metric_logger.meters["gl_loss"].update(gl_loss.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        metric_logger.meters["gan_loss"].update(gan_loss.item(), n=batch_size)


# =============================================================================
# evaluate — ĐÁNH GIÁ accuracy trên tập test (chỉ chạy xuôi, không học)
# =============================================================================
def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()   # chế độ eval (tắt dropout, dùng thống kê BatchNorm đã học)
    metric_logger = new_utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():   # tắt autograd -> nhanh, tiết kiệm RAM
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, _, _, _ = model(image)   # chỉ lấy logits (bỏ 3 đầu ra distill bằng _)
            loss = criterion(output, target)

            acc1, acc5 = new_utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = new_utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


# =============================================================================
# load_data & tiện ích — NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU ImageNet (boilerplate torchvision)
# -----------------------------------------------------------------------------
# Phần này chuẩn của torchvision: đọc ảnh từ thư mục, áp augment (train) / resize-crop
# (test), tạo sampler cho đa GPU, cache dataset cho nhanh. Không chứa logic CAKD riêng.
# =============================================================================
def _get_cache_path(filepath):
    # Tạo đường dẫn file cache cho dataset (đặt tên theo hash của đường dẫn gốc)
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Nạp dữ liệu train + validation, trả về dataset và sampler tương ứng
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            new_utils.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            new_utils.mkdir(os.path.dirname(cache_path))
            new_utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = new_utils.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            new_utils.mkdir(os.path.dirname(cache_path))
            new_utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


# =============================================================================
# main — KHỞI TẠO TẤT CẢ + VÒNG LẶP HUẤN LUYỆN
# =============================================================================
def main(args):
    if args.output_dir:
        new_utils.mkdir(args.output_dir)      # tạo thư mục lưu kết quả

    new_utils.init_distributed_mode(args)     # thiết lập train đa GPU (nếu có)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    # >>> TẠO BỘ BA MODEL CỦA CAKD <<<
    model = torchvision.models.resnet50_cakd(num_classes=num_classes)   # STUDENT (ResNet-50 độ thêm)
    teacher = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)   # TEACHER (ViT pretrain)
    # DISCRIMINATOR: input_nc=1 (attention map 1 kênh), ndf=8 (nhẹ), 3 lớp
    discriminator = new_utils.NLayerDiscriminator(input_nc=1, ndf=8, n_layers=3)
    model.to(device)
    teacher.to(device)
    discriminator.to(device)

    if args.distributed and args.sync_bn:
        # Đồng bộ BatchNorm giữa các GPU (khi batch mỗi GPU nhỏ)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    # >>> 3 HÀM LOSS <<<
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)   # cls_loss (phân loại)
    mse_criterion = nn.MSELoss()                                            # pca_loss & gl_loss (khớp đặc trưng)
    gan_criterion = new_utils.GANLoss().to(device)                          # gan_loss (đối kháng)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = new_utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    # optimizer RIÊNG cho discriminator, learning rate nhỏ hơn 100 lần (0.01*lr) -> discriminator
    # học chậm hơn student, tránh nó "quá mạnh" làm hỏng cân bằng đối kháng.
    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.01*args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov="nesterov" in opt_name)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None   # bộ scale gradient cho mixed-precision (AMP)

    # ----- Bộ điều chỉnh learning rate theo epoch (mỗi optimizer 1 bộ, tên có tiền tố d_ là cho discriminator) -----
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        d_main_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
        d_main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0.01*args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        d_main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
            d_warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                d_optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
            d_warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                d_optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
        d_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            d_optimizer, schedulers=[d_warmup_lr_scheduler, d_main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
        d_lr_scheduler = d_main_lr_scheduler

    # Giữ tham chiếu tới model "trần" (chưa bọc DDP) để lưu/nạp trọng số cho gọn
    model_without_ddp = model
    teacher_without_ddp = teacher
    discriminator_without_ddp = discriminator
    if args.distributed:
        # Bọc DistributedDataParallel để chạy song song nhiều GPU. find_unused_parameters=True vì
        # student có nhánh distill (pca_proj/gl_proj) không phải batch nào cũng dùng hết -> tránh lỗi DDP.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu],find_unused_parameters=True)
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
        model_without_ddp = model.module
        teacher_without_ddp = teacher.module
        discriminator_without_ddp = discriminator.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = new_utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    # ===== VÒNG LẶP HUẤN LUYỆN CHÍNH: lặp qua từng epoch =====
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)   # đổi cách trộn dữ liệu mỗi epoch (đồng bộ giữa các GPU)
        # Huấn luyện 1 epoch (chạy toàn bộ logic loss & GAN ở train_one_epoch phía trên)
        train_one_epoch(model, discriminator, teacher, mse_criterion, gan_criterion, criterion, optimizer, d_optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()      # giảm learning rate của student theo lịch
        d_lr_scheduler.step()    # giảm learning rate của discriminator theo lịch
        evaluate(model, criterion, data_loader_test, device=device)   # đánh giá accuracy sau epoch
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if epoch % 10 == 0:
                new_utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            new_utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


# =============================================================================
# get_args_parser — KHAI BÁO THAM SỐ DÒNG LỆNH (boilerplate argparse)
# -----------------------------------------------------------------------------
# Toàn bộ các cờ để chỉnh khi chạy: --data-path, --batch-size, --lr, --epochs,
# --amp (mixed precision), --distributed... Xem file experiments/run_cakd.sh để
# biết lệnh chạy thực tế truyền cờ nào. Không chứa logic mạng.
# =============================================================================
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datassd2/classification/imagenet/", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    return parser


# Điểm bắt đầu khi chạy `python dist_train_cakd.py ...`: đọc tham số dòng lệnh rồi gọi main()
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
