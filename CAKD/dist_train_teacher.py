# =============================================================================
# GD1 — Fine-tune teacher ViT-B/16 (pretrain ImageNet) xuong 3 lop glass/paper/plastic.
# Vi sao: teacher goc xuat 1000 logits -> khong the distill logits cho student 3 lop
# (gl_loss se vo shape). Fine-tune teacher xuong 3 lop truoc, luu teacher_3cls.pth,
# roi dist_train_cakd.py nap teacher nay de distill.
#
# Chi dung CrossEntropy tren out[0] (forward ViT do tra 4-tuple, phan tu 0 = logits).
# Chay 1 GPU:  torchrun --nproc_per_node=1 dist_train_teacher.py --data-path <dir> ...
# =============================================================================
import datetime
import os
import time

import new_utils
import torch
import torch.utils.data
from torch import nn

from dist_train_cakd import load_data, _append_history  # tai su dung (DRY)
from models.vit_cakd import build_teacher   # ViT-B/16 3 lop, port torch 2.x


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, scaler=None):
    model.train()
    metric_logger = new_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", new_utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(image)[0]  # ViT do tra 4-tuple -> lay logits
            loss = criterion(logits, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        topk = (1, min(3, logits.shape[1]))
        acc1 = new_utils.accuracy(logits, target, topk=topk)[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=image.shape[0])
    return metric_logger  # để main ghi lịch sử train (vẽ biểu đồ)


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = new_utils.MetricLogger(delimiter="  ")
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, "Test:"):
            image, target = image.to(device), target.to(device)
            logits = model(image)[0]
            loss = criterion(logits, target)
            topk = (1, min(3, logits.shape[1]))
            acc1 = new_utils.accuracy(logits, target, topk=topk)[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=image.shape[0])
    metric_logger.synchronize_between_processes()
    print(f"Test Acc@1 {metric_logger.acc1.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def main(args):
    if args.output_dir:
        new_utils.mkdir(args.output_dir)
    new_utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    num_classes = len(dataset.classes)
    print("Classes:", dataset.classes)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.workers, pin_memory=True)

    model = build_teacher(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device)
        return

    print("Start teacher fine-tune")
    start = time.time()
    best = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_ml = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, scaler)
        lr_scheduler.step()
        acc = evaluate(model, criterion, data_loader_test, device)
        _append_history(args, epoch, train_ml, acc, fname="history_teacher.json")
        if args.output_dir:
            ckpt = {"model": model_without_ddp.state_dict(), "epoch": epoch,
                    "classes": dataset.classes, "args": args}
            new_utils.save_on_master(ckpt, os.path.join(args.output_dir, "teacher_3cls.pth"))
            if acc > best:
                best = acc
                new_utils.save_on_master(ckpt, os.path.join(args.output_dir, "teacher_3cls_best.pth"))
    dur = datetime.timedelta(seconds=int(time.time() - start))
    print(f"Teacher fine-tune xong sau {dur}, best acc@1={best:.3f}")


def get_args_parser(add_help=True):
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune ViT teacher xuong N lop", add_help=add_help)
    p.add_argument("--data-path", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("-b", "--batch-size", default=32, type=int)
    p.add_argument("--epochs", default=15, type=int)
    p.add_argument("-j", "--workers", default=2, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--wd", "--weight-decay", default=0.05, type=float, dest="weight_decay")
    p.add_argument("--label-smoothing", default=0.1, type=float)
    p.add_argument("--print-freq", default=10, type=int)
    p.add_argument("--output-dir", default="/kaggle/working", type=str)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--test-only", action="store_true")
    # cac co load_data cua dist_train_cakd can (dat mac dinh khop)
    p.add_argument("--interpolation", default="bilinear", type=str)
    p.add_argument("--val-resize-size", default=256, type=int)
    p.add_argument("--val-crop-size", default=224, type=int)
    p.add_argument("--train-crop-size", default=224, type=int)
    p.add_argument("--cache-dataset", action="store_true")
    p.add_argument("--auto-augment", default=None, type=str)
    p.add_argument("--random-erase", default=0.0, type=float)
    p.add_argument("--ra-magnitude", default=9, type=int)
    p.add_argument("--augmix-severity", default=3, type=int)
    p.add_argument("--ra-sampler", action="store_true")
    p.add_argument("--ra-reps", default=3, type=int)
    p.add_argument("--weights", default=None, type=str)
    p.add_argument("--world-size", default=1, type=int)
    p.add_argument("--dist-url", default="env://", type=str)
    return p


if __name__ == "__main__":
    main(get_args_parser().parse_args())
